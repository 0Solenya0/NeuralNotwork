import torch
from torch import nn
import random

from calculator.generate_data import generate_simple_expression

VOCAB = "0123456789+()= "
VOCAB_SIZE = len("0123456789+()= ")
CONTEXT_SIZE = 32
EMBEDDING_SIZE = 16


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, head_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size

        self.query = nn.Linear(n_embd, head_size * num_heads)
        self.key = nn.Linear(n_embd, head_size * num_heads)
        self.value = nn.Linear(n_embd, head_size * num_heads)
        self.register_buffer('mask', torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)))

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(-1, self.num_heads, x.shape[1], self.head_size)
        k = k.view(-1, self.num_heads, x.shape[1], self.head_size)
        v = v.view(-1, self.num_heads, x.shape[1], self.head_size)

        ws = q @ k.transpose(-2, -1) * self.head_size**-0.5  # relation between nodes
        ws.masked_fill(self.mask[:x.shape[1], :x.shape[1]] == 0, float('-inf'))  # mask out the future
        ws = torch.softmax(ws, dim=-1)  # (bs, num_heads, x.shape[1], x.shape[1])
        ret = ws @ v  # (bs, num_heads, x.shape[1], head_size)

        return ret.transpose(1, 2).contiguous().view(-1, x.shape[1], self.num_heads * self.head_size)


class DecoderBlock(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.attn = MultiHeadSelfAttention(n_embd, 8, 2)
        self.mlp = nn.Sequential(
            nn.Linear(16, 4 * 16),
            nn.GELU(),
            nn.Linear(4 * 16, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CalculatorTransformer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.position_embedding = nn.Embedding(CONTEXT_SIZE, EMBEDDING_SIZE)
        self.blocks = nn.Sequential(*[DecoderBlock(EMBEDDING_SIZE) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(EMBEDDING_SIZE)
        self.head = nn.Linear(EMBEDDING_SIZE, VOCAB_SIZE)

    def forward(self, x):
        x = self.token_embedding(x) + self.position_embedding(torch.arange(x.shape[1]))
        x = self.blocks(x)
        x = self.ln(x)
        x = self.head(x)
        return x

    def generate(self, idx):
        self.eval()
        for i in range(32):
            logits = self(idx[0, -CONTEXT_SIZE:].view(1, -1))
            #print(logits[:, -1, :].shape)
            idx_next = logits[:, -1, :].argmax(-1)
            idx = torch.cat((idx, idx_next.view(1, 1)), dim=1)
        self.train()
        return idx


model = CalculatorTransformer(2)


def get_batch(size=32):
    x = []
    for _ in range(size):
        t = [VOCAB.find(a) for a in generate_simple_expression()]
        while len(t) < CONTEXT_SIZE + 1:
            t.append(VOCAB.find(' '))
        x.append(t)
    x = torch.tensor(x)
    return x[:, :-1], x[:, 1:]


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i in range(20000):
    x, y = get_batch()
    out = model(x)
    loss = nn.functional.cross_entropy(out.reshape(-1, VOCAB_SIZE), y.reshape(-1))
    model.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
        print(f'loss: {loss.item()}')


encode = lambda t: torch.tensor([VOCAB.find(a) for a in t]).view(1, -1)
decode = lambda t: ''.join([VOCAB[a] for a in t])

decode(model.generate(encode('2001231+312131=')).view(-1).tolist())