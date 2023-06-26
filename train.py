import torch
import torch.optim as optim
import torch.nn as nn
from SimpleTransformer import Transformer

# Input sequence and target sequence have a length of 10, and a dimensionality of 512
seq_length = 10
k = 512  # dimensionality
num_classes = 2  # number of classes, a binary classification for example
depth = 6  # number of layers in the encoder/decoder
batch_size = 32

# Randomly generated data
src = torch.rand(batch_size, seq_length, k)
tgt = torch.rand(batch_size, seq_length, k)
y = torch.randint(0, num_classes, (batch_size, seq_length))  # Randomly generated labels

# Initialize your model, criterion and optimizer
model = Transformer(k, depth, num_classes)
criterion = nn.CrossEntropyLoss()
# Simple adam gradient descent for learning
optimizer = optim.Adam(model.parameters()) 

# Number of epochs
n_epochs = 10

# Training loop
for epoch in range(n_epochs):
    optimizer.zero_grad()
    out = model(src, tgt)
    out = out.view(-1, num_classes)  # reshaping for criterion
    y = y.view(-1)  # reshaping for criterion
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
