import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.scale = embed_size ** 0.5

    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ V
        return out

def generate_causal_mask(seq_len, device):
    return torch.tril(torch.ones(seq_len, seq_len, device=device))

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0

        self.head_dim = embed_size // num_heads
        self.num_heads = num_heads

        self.qkv = nn.Linear(embed_size, embed_size * 3)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        Q, K, V = qkv[0], qkv[1], qkv[2]

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ V

        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, T, C)

        return self.fc(out)

class FeedForward(nn.Module):
    def __init__(self, embed_size, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, embed_size * expansion),
            nn.GELU(),
            nn.Linear(embed_size * expansion, embed_size),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, num_heads)
        self.ff = FeedForward(embed_size)

        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_heads=4, num_layers=2, max_len=256):
        super().__init__()

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_len, max_len))
        )

        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(max_len, embed_size)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_size, num_heads) for _ in range(num_layers)]
        )

        self.ln = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        B, T = x.size()

        positions = torch.arange(0, T, device=x.device)
        positions = positions.unsqueeze(0).expand(B, T)
        x = self.token_emb(x) + self.pos_emb(positions)

        mask = self.causal_mask[:T, :T]

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln(x)
        logits = self.fc_out(x)
        return logits
