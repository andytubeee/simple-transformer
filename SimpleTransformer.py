import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.attention = nn.MultiheadAttention(k, num_heads=4)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k))

    def forward(self, value, key, query, mask=None):
        attention_out, _ = self.attention(query, key, value, attn_mask=mask)
        x = self.norm1(attention_out + query)
        ff_out = self.ff(x)
        out = self.norm2(ff_out + x)
        return out

class Encoder(nn.Module):
    def __init__(self, k, depth):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(k) for _ in range(depth)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, k, depth):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(k) for _ in range(depth)])

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(enc_out, enc_out, x, src_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, k, depth, num_classes):
        super().__init__()
        self.encoder = Encoder(k, depth)
        self.decoder = Decoder(k, depth)
        self.out = nn.Linear(k, num_classes)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        out = self.out(dec_out)
        return out
