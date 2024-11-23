import torch.nn as nn

class AdaptiveAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(AdaptiveAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        attn_output, _ = self.self_attn(src, src, src, key_padding_mask=mask)
        src = src + self.dropout(attn_output)
        src = self.norm(src)
        return src

class AdaptiveTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(AdaptiveTransformerEncoderLayer, self).__init__()
        self.adaptive_attention = AdaptiveAttention(d_model, nhead, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.adaptive_attention(src, mask=src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src