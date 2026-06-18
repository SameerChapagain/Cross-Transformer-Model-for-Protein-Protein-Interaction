import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os,sys

# Adjust these imports to your file names
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from Config.model_config import MODEL_CONFIGS

class InputProjection(nn.Module):
    """
    Projects ESM-2 embeddings (2560-dim) to a smaller transformer dimension.
    Includes layer norm and optional dropout.
    """
    def __init__(self, input_dim=2560, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.projection(x)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding."""
    def __init__(self, dim, max_len=900):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x):
        # x: [batch, heads, seq_len, head_dim]
        seq_len = x.size(-2)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # [1,1,L,D/2]
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x_rot = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return x_rot.flatten(-2)

#Residue level interaction
class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention: Query from one protein, Key/Value from the other.
    """
    def __init__(self, d_model=256, n_heads=8, dropout=0.1, max_len=900, use_rotary=True):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_rotary = use_rotary

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        if self.use_rotary:
            self.rotary = RotaryPositionalEmbedding(self.d_k, max_len=max_len)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key_value, query_mask=None, kv_mask=None):
        batch_size = query.size(0)
        residual = query

        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key_value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(key_value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        if self.use_rotary:
            Q = self.rotary(Q)
            K = self.rotary(K)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if kv_mask is not None:
            kv_mask_expanded = kv_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~kv_mask_expanded, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        output = self.layer_norm(residual + self.dropout(output))
        return output, attn_weights


class TransformerEncoderBlock(nn.Module):
    """Self-attention + FFN."""
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1, max_len=900, use_rotary=True):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_rotary = use_rotary

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        if self.use_rotary:
            self.rotary = RotaryPositionalEmbedding(self.d_k, max_len=max_len)

        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        residual = x

        Q = self.W_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        if self.use_rotary:
            Q = self.rotary(Q)
            K = self.rotary(K)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_expanded, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        attn_out = self.W_o(attn_out)
        x = self.norm1(residual + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class CrossTransformerBlock(nn.Module):
    """
    One block:
    Self-attention → Cross-attention → FFN
    """
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1, max_len=900, use_rotary=True):
        super().__init__()

        self.self_attn_block = TransformerEncoderBlock(
            d_model, n_heads, d_ff, dropout, max_len, use_rotary
        )
        self.cross_attn = MultiHeadCrossAttention(
            d_model, n_heads, dropout, max_len, use_rotary
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, partner, x_mask=None, partner_mask=None):
        x = self.self_attn_block(x, x_mask)
        x, cross_attn_weights = self.cross_attn(x, partner, x_mask, partner_mask)
        ffn_out = self.ffn(x)
        x = self.norm(x + ffn_out)
        return x, cross_attn_weights


class AttentionPooling(nn.Module):
    """
    Attention-based pooling to aggregate residue-level features
    into a single protein-level vector.
    """
    def __init__(self, d_model=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, mask=None):
        attn_scores = self.attention(x).squeeze(-1)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        pooled = (x * attn_weights).sum(dim=1)

        return pooled


class SiameseCrossTransformer(nn.Module):
    """
    Complete Siamese Cross-Transformer for PPI Prediction.
    """

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = {}

        self.input_dim = config.get("input_dim", 2560)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.n_heads = config.get("n_heads", 8)
        self.n_cross_layers = config.get("n_cross_layers", 2)
        self.d_ff = config.get("d_ff", 1024)
        self.dropout = config.get("dropout", 0.1)
        self.max_len = config.get("max_len", 900)
        self.use_attention_pool = config.get("use_attention_pool", True)
        self.use_rotary = config.get("use_rotary", True)
        self.use_sigmoid_output = config.get("use_sigmoid_output", True)

        self.input_projection = InputProjection(
            self.input_dim, self.hidden_dim, self.dropout
        )

        self.cross_transformer_blocks = nn.ModuleList([
            CrossTransformerBlock(
                self.hidden_dim,
                self.n_heads,
                self.d_ff,
                self.dropout,
                self.max_len,
                self.use_rotary
            )
            for _ in range(self.n_cross_layers)
        ])

        if self.use_attention_pool:
            self.pooling = AttentionPooling(self.hidden_dim)

        fusion_dim = self.hidden_dim * 4

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode(self, emb, mask):
        x = self.input_projection(emb)
        return x

    def _pool(self, x, mask):
        if self.use_attention_pool:
            return self.pooling(x, mask)
        else:
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                x_masked = x * mask_expanded
                pooled = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
            return pooled

    def forward(self, emb_a, emb_b, mask_a=None, mask_b=None):
        h_a = self._encode(emb_a, mask_a)
        h_b = self._encode(emb_b, mask_b)

        for cross_block in self.cross_transformer_blocks:
            h_a_new, _ = cross_block(h_a, h_b, mask_a, mask_b)
            h_b_new, _ = cross_block(h_b, h_a, mask_b, mask_a)

            h_a = h_a_new
            h_b = h_b_new

        pooled_a = self._pool(h_a, mask_a)
        pooled_b = self._pool(h_b, mask_b)

        fusion = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        logits = self.classifier(fusion)

        if self.use_sigmoid_output:
            return torch.sigmoid(logits)
        return logits

    def predict_proba(self, emb_a, emb_b, mask_a=None, mask_b=None):
        logits_or_probs = self.forward(emb_a, emb_b, mask_a, mask_b)
        if self.use_sigmoid_output:
            return logits_or_probs
        return torch.sigmoid(logits_or_probs)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

if __name__ == "__main__":
    config = MODEL_CONFIGS["small"]
    model = SiameseCrossTransformer(config)
    print(f"Model parameters: {model.count_parameters():,}")

    batch_size = 12
    max_len = config["max_len"]

    emb_a = torch.randn(batch_size, max_len, config["input_dim"])
    emb_b = torch.randn(batch_size, max_len, config["input_dim"])
    mask_a = torch.ones(batch_size, max_len, dtype=torch.bool)
    mask_b = torch.ones(batch_size, max_len, dtype=torch.bool)

    output = model(emb_a, emb_b, mask_a, mask_b)
    print(f"Output shape: {output.shape}")  