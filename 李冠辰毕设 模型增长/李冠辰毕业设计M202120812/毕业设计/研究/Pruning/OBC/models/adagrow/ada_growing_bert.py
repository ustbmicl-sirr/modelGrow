import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -float("inf"))
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class BertLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, forward_expansion * embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.norm1(attention + query)
        forward = self.ff(x)
        out = self.norm2(forward + x)
        return out


class BertForClassification(nn.Module):
    def __init__(self, depth, heads, dim_head, num_classes, vocab_size, pad_token_id=0):
        super().__init__()
        self.embed_size = dim_head * heads
        self.pad_token_id = pad_token_id
        self.word_embeddings = nn.Embedding(vocab_size, self.embed_size)
        self.layers = nn.ModuleList([BertLayer(self.embed_size, heads, 4) for _ in range(depth)])
        self.fc_out = nn.Linear(self.embed_size, num_classes)

    def make_attention_mask(self, input_ids):
        mask = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask

    def forward(self, x):
        out = self.word_embeddings(x)
        attention_mask = self.make_attention_mask(x)
        for layer in self.layers:
            out = layer(out, out, out, attention_mask)
        out = out.mean(dim=1)
        return self.fc_out(out)


def get_ada_growing_bert(depth=2, heads=6, num_classes=10, vocab_size=30522):
    return BertForClassification(depth=depth, heads=heads, dim_head=64, num_classes=num_classes, vocab_size=vocab_size)
