import os              # File/folder operations
import random          # Random sampling for generating negative protein pairs
import gzip            # Decompressing downloaded .gz STRING files
import shutil          # used for unzipping: copying gz stream to output file
import logging         # Printing clean progress/info logs to terminal
import subprocess      # Running commands (MMseqs2 clustering) from within Python
import json            # Parsing JSON response from STRING API (to get current STRING version)
import requests        # HTTP requests to STRING API (fetch current version + stable endpoint)
import pandas as pd    # reading/writing TSV, filtering pairs, handling clusters tables
from tqdm import tqdm  # Progress bars for long loops
from Bio import SeqIO  # BioPython FASTA read/write: parse sequences and write filtered FASTA files
import wget            # Downloading STRING files from direct URLs
from pandarallel import pandarallel  # Parallelized pandas apply for faster negative-pair filtering

# ============================================================
# CONFIG (edit values here)
# These parameters control how the dataset is created.
# ============================================================
CONFIG = {
     # STRING taxonomy ID (Human = 9606)
    "species": "9606",

    # Keep only protein sequences within this length range (aa)
    "min_length": 50,
    "max_length": 1000,

    # Limit number of positive pairs (None = keep all)
    "max_positive_pairs": None,

    # STRING confidence thresholds (0–1000)
    "combined_score": 900,
    "experimental_score": None,

    # If True, do not filter sequences by length (keeps original FASTA)
    "not_remove_long_short_proteins": False,
    "log_level": "INFO",
}

# Base download URL for STRING flat-files
DOWNLOAD_LINK_STRING = "https://stringdb-downloads.org/download/"

# Get stable api and current STRING version
def get_string_url():
    request_url = "/".join(["https://string-db.org/api", "json", "version"])
    response = requests.post(request_url)
    info = json.loads(response.text)[0]
    return "/".join([info["stable_address"], "api"]), info["string_version"]


def _count_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)

# Dataset creation pipeline
class STRINGDatasetCreation:
    def __init__(self, interactions_file, 
                sequences_file, 
                min_length, 
                max_length, 
                species,
                max_positive_pairs, 
                combined_score, 
                experimental_score, 
                not_remove_long_short_proteins):

        self.interactions_file = interactions_file
        self.sequences_file = sequences_file
        self.min_length = min_length
        self.max_length = max_length
        self.species = "custom" if species is None else species
        self.max_positive_pairs = max_positive_pairs
        self.combined_score = combined_score
        self.experimental_score = experimental_score
        self.intermediate_file = "interactions_intermediate.tmp"

        if not not_remove_long_short_proteins:
            self._filter_fasta_by_length()
        self._select_interactions()

    def _filter_fasta_by_length(self):
        logging.info("Filtering FASTA by length [%d, %d] ...", self.min_length, self.max_length)
        with open("seqs.tmp", "w") as f:
            for record in tqdm(SeqIO.parse(self.sequences_file, "fasta")):
                if self.min_length <= len(record.seq) <= self.max_length:
                    record.description = ""
                    record.name = ""
                    SeqIO.write(record, f, "fasta")
        os.replace("seqs.tmp", self.sequences_file)


    def _select_interactions(self):
        if os.path.isfile(self.intermediate_file):
            logging.info("Intermediate file exists (%s). Skipping selection.", self.intermediate_file)
            return

        # ------------------------------------------------------------
        # 1) Run clustering on all sequences (needed to remove redundancy)
        # ------------------------------------------------------------
        if not os.path.isfile("clusters.tsv"):
            logging.info("clusters.tsv not found. Running MMseqs2 clustering (40%% identity).")
            logging.info("This may take time depending on species FASTA size.")
            commands = "; ".join([
                "mkdir mmseqDBs",
                f"mmseqs createdb {self.sequences_file} mmseqDBs/DB",
                "mmseqs cluster mmseqDBs/DB mmseqDBs/clusterDB tmp --min-seq-id 0.4 --alignment-mode 3 --cov-mode 1 --threads 8",
                "mmseqs createtsv mmseqDBs/DB mmseqDBs/DB mmseqDBs/clusterDB clusters.tsv",
                "rm -r mmseqDBs tmp"
            ])
            subprocess.run(commands, shell=True)
            logging.info("MMseqs2 clustering done -> clusters.tsv created.")
        clusters = (
            pd.read_csv("clusters.tsv", sep="\t", header=None, names=["cluster", "protein"])
            .set_index("protein")["cluster"].to_dict()
        )

        with open(self.interactions_file, "rb") as f:
            total_lines = sum(chunk.count(b"\n") for chunk in _count_generator(f.raw.read)) + 1

        logging.info("Selecting interactions (combined_score >= %d) ...", self.combined_score)
        seen_cluster_pairs = set()

        with open(self.intermediate_file, "w") as out_f, open(self.interactions_file) as in_f:
            out_f.write("\t".join(in_f.readline().strip().split(" ")) + "\n")

            for line in tqdm(in_f, total=total_lines):
                parts = line.strip().split(" ")
                if len(parts) < 4:
                    continue
                # Species filter (STRING ids start with taxon prefix like "9606.")
                if self.species != "custom":
                    if not parts[0].startswith(self.species) or not parts[1].startswith(self.species):
                        continue

                if self.experimental_score is not None and int(parts[3]) < self.experimental_score:
                    continue

                # Original selection logic:
                # - int(parts[2]) == 0 => exclude homology-transferred interactions
                # - int(parts[-1]) >= combined_score threshold
                if int(parts[2]) == 0 and int(parts[-1]) >= self.combined_score:
                    if parts[0] not in clusters or parts[1] not in clusters:
                        continue
                    # Define cluster-pair for redundancy filtering
                    c1, c2 = sorted((clusters[parts[0]], clusters[parts[1]]))
                    if (c1, c2) not in seen_cluster_pairs:
                        seen_cluster_pairs.add((c1, c2))
                        out_f.write("\t".join(parts) + "\n")

    def final_preprocessing_positives(self):
        logging.info("Final preprocessing positives started...")
        data = pd.read_csv(self.intermediate_file, sep="\t")[["protein1", "protein2", "combined_score"]]

        if self.max_positive_pairs is not None:
            data = data.sort_values("combined_score", ascending=False).iloc[:self.max_positive_pairs]

        # Label positives as 1
        data["combined_score"] = 1
        tmp_pairs = f"protein.pairs_{self.species}.tsv.tmp"
        data.to_csv(tmp_pairs, sep="\t", index=False)

        proteins = set(data["protein1"]).union(data["protein2"])
        with open(f"sequences_{self.species}.fasta", "w") as f:
            for record in tqdm(SeqIO.parse(self.sequences_file, "fasta")):
                if record.id in proteins:
                    SeqIO.write(record, f, "fasta")
        logging.info("Final preprocessing positives done.")
    
    def create_negatives(self):
        """
        Generates negative pairs in 1:10 ratio (positive:negative) and writes final output:
          protein.pairs_<species>.tsv

        Note:
        - This version removes negative pairs already present in positives (both directions).
        - Clusters_preprocessed.tsv is produced by MMseqs2 clustering on the filtered FASTA.
        """
        logging.info("Negative pair generation started...")
        if not os.path.isfile("clusters_preprocessed.tsv"):
            logging.info("clusters_preprocessed.tsv not found. Running MMseqs2 on filtered FASTA...")
            commands = "; ".join([
                "mkdir mmseqDBs",
                f"mmseqs createdb sequences_{self.species}.fasta mmseqDBs/DB",
                "mmseqs cluster mmseqDBs/DB mmseqDBs/clusterDB tmp --min-seq-id 0.4 --alignment-mode 3 --cov-mode 1 --threads 8",
                "mmseqs createtsv mmseqDBs/DB mmseqDBs/DB mmseqDBs/clusterDB clusters_preprocessed.tsv",
                "rm -r mmseqDBs tmp"
            ])
            subprocess.run(commands, shell=True)
            logging.info("MMseqs2 on filtered FASTA done -> clusters_preprocessed.tsv created.")

        clusters = (
            pd.read_csv("clusters_preprocessed.tsv", sep="\t", header=None, names=["cluster", "protein"])
            .set_index("protein")["cluster"].to_dict()
        )

        interactions = pd.read_csv(f"protein.pairs_{self.species}.tsv.tmp", sep="\t")
        proteins = list(clusters.keys())

        #Initialize parallel apply
        pandarallel.initialize(progress_bar=True)

        neg = pd.DataFrame({
            "protein1": random.choices(proteins, k=len(interactions) * 12),
            "protein2": random.choices(proteins, k=len(interactions) * 12),
            "combined_score": 0,
        })

        # Normalize (protein1, protein2) ordering -> makes duplicates easy to remove
        logging.info("Normalizing pair order and removing duplicates...")
        neg["protein1"], neg["protein2"] = zip(*neg.parallel_apply(
            lambda x: (x["protein1"], x["protein2"]) if x["protein1"] < x["protein2"] else (x["protein2"], x["protein1"]),
            axis=1
        ))
        neg = neg.drop_duplicates()
        logging.info("Duplicates removed!")

        neg = neg[
            ~neg.parallel_apply(lambda x: len(interactions[
                (interactions["protein1"] == x["protein1"]) & (interactions["protein2"] == x["protein2"])
            ]) > 0, axis=1)
        ]
        neg = neg[
            ~neg.parallel_apply(lambda x: len(interactions[
                (interactions["protein1"] == x["protein2"]) & (interactions["protein2"] == x["protein1"])
            ]) > 0, axis=1)
        ]

        neg = neg.iloc[:len(interactions) * 10]

        # Final dataset: positives + negatives
        final_pairs = pd.concat([interactions, neg], ignore_index=True)
        final_pairs.to_csv(f"protein.pairs_{self.species}.tsv", sep="\t", index=False, header=False)

        # Cleanup temporary files (optional but similar to original)
        logging.info("Cleaning up temporary files...")
        os.remove(f"protein.pairs_{self.species}.tsv.tmp")
        os.remove(self.intermediate_file)
        os.remove("clusters.tsv")
        os.remove("clusters_preprocessed.tsv")
        logging.info("Cleanup complete.")


def download_string_files(species):
    _, version = get_string_url()
    logging.info("STRING version detected: %s", version)

    links = f"{species}.protein.physical.links.full.v{version}.txt"
    seqs = f"{species}.protein.sequences.v{version}.fa"

    logging.info("Downloading interactions file: %s.gz", links)
    wget.download(f"{DOWNLOAD_LINK_STRING}protein.physical.links.full.v{version}/{links}.gz", out=links + ".gz")

    logging.info("Downloading sequences file: %s.gz", seqs)
    wget.download(f"{DOWNLOAD_LINK_STRING}protein.sequences.v{version}/{seqs}.gz", out=seqs + ".gz")

    logging.info("Unzipping downloaded files...")
    with gzip.open(links + ".gz", "rb") as f_in, open(links, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    with gzip.open(seqs + ".gz", "rb") as f_in, open(seqs, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    os.remove(links + ".gz")
    os.remove(seqs + ".gz")

    logging.info("Download + unzip complete.")
    return links, seqs


def run():
    """
    Entry point: runs the full dataset creation pipeline using CONFIG.
    """
    logging.basicConfig(level=getattr(logging, CONFIG["log_level"], logging.INFO))

    interactions, sequences = download_string_files(CONFIG["species"])

    dataset = STRINGDatasetCreation(
        interactions, sequences,
        CONFIG["min_length"], 
        CONFIG["max_length"],
        CONFIG["species"], 
        CONFIG["max_positive_pairs"],
        CONFIG["combined_score"], 
        CONFIG["experimental_score"],
        CONFIG["not_remove_long_short_proteins"]
    )

    dataset.final_preprocessing_positives()
    dataset.create_negatives()

     # Remove raw downloaded files 
    logging.info("Removing downloaded raw STRING files...")
    os.remove(interactions)
    os.remove(sequences)

    logging.info("Removed downloaded raw STRING files...")
    logging.info("Dataset creation finished successfully ✅")


if __name__ == "__main__":
    run()
