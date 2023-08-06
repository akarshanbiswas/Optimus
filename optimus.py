import torch
import math
import numpy as np


class PositionalEncoding(torch.nn.Module):
    def __init__(self, embed_size, max_len=2049):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class OptimusNorm(torch.nn.Module):
    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super.__init__()
        self.weight = torch.nn.Parameter(torch.ones(embed_dim))
        self.eps = eps

    def _n(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._n(x.float()).type_as(x)
        x = x * self.weight
        return x


class OptimusAttention(torch.nn.Module):
    def __init__(self, embed_size: int, num_heads: int):
        super(OptimusAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.qkv_proj = torch.nn.Linear(embed_size, 3 * embed_size, bias=False)
        self.out_proj = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        torch.nn.init.uniform_(self.qkv_proj.weight.data, -initrange, initrange)
        torch.nn.init.uniform_(self.out_proj.weight.data, -initrange, initrange)

    # def softmax_1(self, x):
    #     return torch.exp(x) / (1 + sum(torch.exp(x)))

    # def compute_attn_scores(self, q: torch.Tensor,
    #                         k: torch.Tensor,
    #                         v: torch.Tensor) -> torch.Tensor:
    #     # Scaled dot product attention
    #     sim_scores = q @ k.transpose(-2, -1) / torch.sqrt(
    #         torch.tensor(q.size(-1)).float())

    #     mask = torch.triu(
    #         torch.ones(
    #             sim_scores.size(-2), sim_scores.size(-1)),
    #         diagonal=1).bool().to(sim_scores.device)
    #     sim_scores = sim_scores.masked_fill(mask == 1, float('-inf'))
    #     # w = torch.nn.functional.softmax(sim_scores, dim=-1)
    #     w = self.softmax_1(sim_scores)
    #     wv = w @ v
    #     return wv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        # [Batch, Head, SeqLen, Dims]
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        # # Use flash attention
        scores = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )
        # scores = self.compute_attn_scores(q, k, v)
        scores = scores.permute(0, 2, 1, 3)
        scores = scores.reshape(batch_size, seq_length, self.embed_size)
        output = self.out_proj(scores)

        return output


class Optimuslayer(torch.nn.Module):
    def __init__(self, embed_size: int, nheads: int, dropout: float):
        super(Optimuslayer, self).__init__()
        self.embed_size = embed_size
        self.nheads = nheads
        self.fw_dim = 4 * self.embed_size
        self.attn = OptimusAttention(self.embed_size, self.nheads)
        self.feedfw = torch.nn.Sequential(
            torch.nn.Linear(self.embed_size, self.fw_dim, bias=False),
            torch.nn.Linear(self.fw_dim, self.embed_size, bias=False),
            torch.nn.Linear(self.embed_size, self.embed_size, bias=False),
            torch.nn.SiLU(),
        )
        self.norm1 = OptimusNorm(self.embed_size)
        self.norm2 = OptimusNorm(self.embed_size)
        self.drop = torch.nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        for layer in self.feedfw:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.uniform_(layer.weight, -init_range, init_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp1 = self.norm1(x)

        attn_out = self.attn(inp1)

        x = x + self.drop(attn_out)

        inp2 = self.norm2(x)
        output2 = self.feedfw(inp2)
        x = x + self.drop(output2)

        return x


class Optimus(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        nheads: int,
        nlayers: int,
        maxlen: int,
        dropout: float = 0.0,
    ):
        """
        vocab_size: Total vocabulary size
        embed_size: Hidden dimension size
        nheads: number of attention heads
        nlayers: Number of transformer layer blocks
        maxlen: contextsize for PositionalEncoding
        dropout: dropout probability, 0.1
        Example:
        model = Optimus(100, 64, 8, 4, 100, 0.1)
        logits = model(input_tensor)
        """
        super(Optimus, self).__init__()
        if embed_size % nheads != 0:
            raise ValueError(
                "Embedding dimension should be divisible by number of heads"
            )
        self.embed_size = embed_size
        self.nlayers = nlayers
        self.nheads = nheads
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=self.embed_size, padding_idx=0
        )
        self.poe = PositionalEncoding(self.embed_size, maxlen)
        self.layers = torch.nn.ModuleList(
            [
                Optimuslayer(self.embed_size, self.nheads, dropout)
                for _ in range(self.nlayers)
            ]
        )
        self.vocab_classifier = torch.nn.Linear(embed_size, vocab_size, bias=False)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        torch.nn.init.uniform_(self.embedding.weight.data, -initrange, initrange)
        torch.nn.init.uniform_(self.vocab_classifier.weight.data, -initrange, initrange)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.embed_size)
        x = self.poe(x)
        for layer in self.layers:
            x = layer(x)
        x = self.vocab_classifier(x)

        return x


class OptimusWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_steps):
        self.warmup = warmup
        self.max_num_steps = max_steps
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_steps))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
