import torch

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

type(text) == str
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

encoded_text = [char_to_idx[ch] for ch in text]

sequence_length = 100
inputs = []
targets = []

for i in range(0, len(encoded_text) - sequence_length):
    inputs.append(encoded_text[i:i + sequence_length])
    targets.append(encoded_text[i + 1:i + sequence_length + 1])

inputs = torch.tensor(inputs, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

batch_size = 64

dataset = torch.utils.data.TensorDataset(inputs, targets)

loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True
)

print("Input:", "".join(idx_to_char[i.item()] for i in inputs[0]))
print("Target:", "".join(idx_to_char[i.item()] for i in targets[0]))

# --- training implementation using model.py ---
import torch.nn as nn
from model import CharRNN

# hyperparameters
embed_size = 128
hidden_size = 256
num_layers = 2
dropout = 0.0
lr = 1e-3
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CharRNN(vocab_size, embed_size, hidden_size, num_layers, dropout).to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)
crit = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)               # (batch, seq)
        opt.zero_grad()
        logits, _ = model(xb)                               # (batch, seq, vocab)
        loss = crit(logits.view(-1, vocab_size), yb.view(-1))
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    avg = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{epochs} loss: {avg:.4f}")

# save checkpoint with vocab and hyperparams
torch.save({
    "model_state": model.state_dict(),
    "vocab_size": vocab_size,
    "embed_size": embed_size,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "char_to_idx": char_to_idx,
    "idx_to_char": idx_to_char,
    "sequence_length": sequence_length,
}, "checkpoint.pth")