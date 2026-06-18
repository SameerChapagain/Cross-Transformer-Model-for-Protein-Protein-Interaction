from pathlib import Path
import torch
from esm import FastaBatchedDataset, pretrained

def run():
    fasta_file = "/kaggle/input/fasta-splits/sequences_9606.part_001.fasta" #kaggle input directory
    device = "gpu"
    output_dir_esm=Path('esm2_embs_3B')
    repr_layers_esm = [-1] #Last esm2 model layer
    model, alphabet = pretrained.load_model_and_alphabet("esm2_t36_3B_UR50D")
    model.eval()

    if device == 'gpu':
        model = model.cuda()
        print("Transferred the ESM2 model to GPU")
    elif device == 'mps':
        model = model.to('mps')
        print("Transferred the ESM2 model to MPS")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(900), batch_sampler=batches
    )
    print(f"Read {fasta_file} with {len(dataset)} sequences")

    output_dir_esm.mkdir(parents=True, exist_ok=True)

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers_esm)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers_esm]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if device == 'gpu':
                toks = toks.to(device="cuda", non_blocking=True)
            elif device == 'mps':
                toks = toks.to(device="mps", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                output_file_esm = output_dir_esm / f"{label}.pt"
                output_file_esm.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(900, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                result["representations"] = {
                    layer: t[i, 1: truncate_len + 1].clone()
                    for layer, t in representations.items()
                }

                torch.save(
                    result,
                    output_file_esm,
                )
    print("✅ ESM2 embedding generation completed successfully.")
    print(f"📁 Embeddings saved in: {output_dir_esm}")
run()
