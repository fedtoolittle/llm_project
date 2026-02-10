import torch
import torch.nn as nn


class CharRNN(nn.Module):
    """Small char-level RNN: Embedding -> GRU -> Linear (vocab logits).

    GRU is used instead of LSTM because some backends (DirectML) may not
    support fused LSTM kernels.
    """
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Implement GRU cell with explicit linear layers (no fused kernel).
        # This avoids calling backend fused ops that may be unsupported and
        # instead uses basic linear/activation ops which have wider support.
        self.linear_ih = nn.Linear(embed_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x, hidden=None):
        """Forward pass.

        Args:
            x: LongTensor of shape (batch, seq_len)
            hidden: optional h0 for GRU

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: GRU hidden tensor
        """
        emb = self.embed(x)  # (batch, seq, embed)
        batch_size, seq_len, _ = emb.size()
        if hidden is None:
            hidden = self.init_hidden(batch_size, device=emb.device)

        outputs = []
        h = hidden
        for t in range(seq_len):
            x_t = emb[:, t, :]
            i_r, i_z, i_n = self.linear_ih(x_t).chunk(3, dim=-1)
            h_r, h_z, h_n = self.linear_hh(h).chunk(3, dim=-1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = (1 - z) * n + z * h
            outputs.append(h.unsqueeze(1))
        out = torch.cat(outputs, dim=1)  # (batch, seq, hidden)
        logits = self.fc(out)
        return logits, h

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        # For GRUCell we use a hidden of shape (batch, hidden_size)
        h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return h0
