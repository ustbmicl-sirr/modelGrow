import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import init


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


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
        hidden_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(hidden_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
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


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, depth, heads, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        hidden_dim = heads * dim_head
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(depth, heads, dim_head, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(hidden_dim, num_classes)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)
    

def get_rand_growing_vit_patch2_32(depth=1, heads=8, num_classes=10, image_channels=3):
    return ViT(image_size=32, patch_size=2, num_classes=num_classes, depth=depth, heads=heads, channels=image_channels, dim_head=64)


def get_rand_growing_vit_patch4_64(depth=1, heads=8, num_classes=10, image_channels=3):
    return ViT(image_size=64, patch_size=4, num_classes=num_classes, depth=depth, heads=heads, channels=image_channels, dim_head=64)


def get_rand_growing_vit_patch16_224(depth=1, heads=8, num_classes=10, image_channels=3):
    return ViT(image_size=224, patch_size=16, num_classes=num_classes, depth=depth, heads=heads, channels=image_channels, dim_head=64)
