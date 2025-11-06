import math
import torch
from torch import nn
from einops import rearrange
from torch.nn import init


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        hidden_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(hidden_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, depth, heads, dim_head, dropout=0.):
        super().__init__()
        hidden_dim = heads * dim_head
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(hidden_dim, hidden_dim * 4, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class BertForClassification(nn.Module):
    def __init__(self, depth=12, heads=12, dim_head=64, num_classes=3, vocab_size=30000):
        super(BertForClassification, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, heads * dim_head)
        self.pos_encoding = PositionalEncoding(heads * dim_head, 5000)
        self.encoder = Transformer(depth, heads, dim_head)
        self.classifier = nn.Linear(heads * dim_head, num_classes)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        pos_embedding_output = self.pos_encoding(embedding_output)
        encoded_output = self.encoder(pos_embedding_output)
        cls_output = encoded_output[:, 0, :]
        classification_output = self.classifier(cls_output)
        return classification_output


def get_runtime_bert(depth=2, heads=6, num_classes=10, vocab_size=30522):
    return BertForClassification(depth=depth, heads=heads, dim_head=64, num_classes=num_classes, vocab_size=vocab_size)
