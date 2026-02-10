import torch
import torch.nn as nn


class CharRNN(nn.Module):
    """Small char-level RNN: Embedding -> LSTM -> Linear (vocab logits)."""
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x, hidden=None):
        """Forward pass.

        Args:
            x: LongTensor of shape (batch, seq_len)
            hidden: optional tuple (h0, c0) for LSTM

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: LSTM hidden tuple
        """
        emb = self.embed(x)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)
import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x, hidden=None):
        emb = self.embed(x)                 # (batch, seq, embed)
        out, hidden = self.rnn(emb, hidden) # out: (batch, seq, hidden)
        logits = self.fc(out)               # (batch, seq, vocab)
        return logits, hidden

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)