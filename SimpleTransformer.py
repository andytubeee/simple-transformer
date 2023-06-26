import torch
import torch.nn as nn

# Define a transformer block, which forms the basic unit of the transformer architecture
class TransformerBlock(nn.Module):
    def __init__(self, k):
        super().__init__()

        # Multi-head attention mechanism. It allows the model to focus on different parts of the input simultaneously
        self.attention = nn.MultiheadAttention(k, num_heads=4)

        # Layer normalization helps stabilize the inputs to each layer, similar to how batch normalization works
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        # Feed-forward network, which applies linear transformations followed by a non-linear activation (ReLU here)
        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k))

    def forward(self, value, key, query, mask=None):
        # The forward function is how the transformer block processes inputs
        # Attention operation. The output consists of weighted sum of the inputs
        attention_out, _ = self.attention(query, key, value, attn_mask=mask)

        # Apply normalization and residual connection
        x = self.norm1(attention_out + query)

        # Apply the feed-forward network
        ff_out = self.ff(x)

        # Apply the second normalization and residual connection
        out = self.norm2(ff_out + x)

        return out

# The encoder processes the input data using multiple transformer blocks
class Encoder(nn.Module):
    def __init__(self, k, depth):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(k) for _ in range(depth)])

    def forward(self, x, mask=None):
        # Each transformer block in the encoder processes the input
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x

# The decoder generates the output data, also using multiple transformer blocks
class Decoder(nn.Module):
    def __init__(self, k, depth):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(k) for _ in range(depth)])

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # Each transformer block in the decoder processes both the output of the encoder and the input to the decoder
        for layer in self.layers:
            x = layer(enc_out, enc_out, x, src_mask)
        return x

# The complete transformer model consists of an encoder, a decoder, and a final linear layer to produce output
class Transformer(nn.Module):
    def __init__(self, k, depth, num_classes):
        super().__init__()

        # Encoder part of the model
        self.encoder = Encoder(k, depth)

        # Decoder part of the model
        self.decoder = Decoder(k, depth)

        # Final linear layer to produce the output
        self.out = nn.Linear(k, num_classes)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder processes the source (input) sequence
        enc_out = self.encoder(src, src_mask)

        # Decoder uses output of encoder along with target sequence to produce a sequence of vectors
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)

        # Linear layer maps from the sequence of vectors to sequence of output (class probabilities in case of classification)
        out = self.out(dec_out)
        return out
