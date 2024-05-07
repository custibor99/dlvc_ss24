## TODO implement your own ViT in this file
# You can take any existing code from any repository or blog post - it doesn't have to be a huge model
# specify from where you got the code and integrate it into this code repository so that 
# you can run the model with this code

from torch import nn
import torch
import numpy as np

def get_sinusoid_encoding(num_tokens, token_len):
    def get_position_angle_vec(i):
        return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

    sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    res = torch.FloatTensor(sinusoid_table).unsqueeze(0)
    res.requires_grad = False
    return res

class PatchEmbedding(torch.nn.Module):
    def __init__(self, patch_size=8):
        super().__init__()
        self.p = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.flatten = torch.nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        # x -> B c h w
        bs, c, h, w = x.shape
        x = self.unfold(x)

        # Reshaping into the shape we want
        a = x.view(bs, c, self.p, self.p, -1).permute(0, 4, 2, 3, 1)
        return self.flatten(a)


class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def forward(self, x):
        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        x, w = self.attn(q, k, v)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = nn.Sequential(nn.LayerNorm(dim),
                                   nn.Linear(dim, dim),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(dim, dim)
                                   )

    def forward(self, x):
        return self.layer(x)


class VisionTransformerShallow(torch.nn.Module):
    def __init__(self, dropout_rate=0.5, channels=3, n_layers=2, img_size=32, patch_size=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.n_layers = n_layers
        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_size = patch_size
        self.patcher = PatchEmbedding(patch_size=patch_size)
        self.num_tokens = int((32 / patch_size)) ** 2
        self.token_len = patch_size ** 2
        self.positional_encoding = get_sinusoid_encoding(self.num_tokens, self.patch_size ** 2)
        self.positional_encoding = self.positional_encoding.view(1, self.num_tokens, self.patch_size ** 2, 1)
        self.flatten = torch.nn.Flatten(start_dim=2, end_dim=3)
        dim = self.token_len * 3

        self.att1 = Attention(dim, 4, dropout_rate)
        self.ff1 = FeedForward(dim, dropout_rate)

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim * self.num_tokens, 10)
        )

    def forward(self, x):
        x = self.patcher(x)
        x = x + self.positional_encoding
        x = self.flatten(x)
        x = self.att1(x)
        x = self.ff1(x)
        x = self.out(x)
        return x


class VisionTransformerDeep(torch.nn.Module):
    def __init__(self, dropout_rate=0.5, channels=3, n_layers=2, img_size=32, patch_size=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.n_layers = n_layers
        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_size = patch_size
        self.patcher = PatchEmbedding(patch_size=patch_size)
        self.num_tokens = int((32 / patch_size)) ** 2
        self.token_len = patch_size ** 2
        self.positional_encoding = get_sinusoid_encoding(self.num_tokens, self.patch_size ** 2)
        self.positional_encoding = self.positional_encoding.view(1, self.num_tokens, self.patch_size ** 2, 1)
        self.flatten = torch.nn.Flatten(start_dim=2, end_dim=3)
        dim = self.token_len * 3

        self.att1 = Attention(dim, 4, dropout_rate)
        self.ff1 = FeedForward(dim, dropout_rate)

        self.att2 = Attention(dim, 4, dropout_rate)
        self.ff2 = FeedForward(dim, dropout_rate)

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim * self.num_tokens, 10)
        )

    def forward(self, x):
        x = self.patcher(x)
        x = x + self.positional_encoding
        x = self.flatten(x)
        x = self.att1(x)
        x = self.ff1(x)
        x = self.att2(x)
        x = self.ff2(x)
        x = self.out(x)
        return x


class VisionTransformerDeepResidual(torch.nn.Module):
    def __init__(self, dropout_rate=0.5, channels=3, n_layers=2, img_size=32, patch_size=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.n_layers = n_layers
        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_size = patch_size
        self.patcher = PatchEmbedding(patch_size=patch_size)
        self.num_tokens = int((32 / patch_size)) ** 2
        self.token_len = patch_size ** 2
        self.positional_encoding = get_sinusoid_encoding(self.num_tokens, self.patch_size ** 2)
        self.positional_encoding = self.positional_encoding.view(1, self.num_tokens, self.patch_size ** 2, 1)
        self.flatten = torch.nn.Flatten(start_dim=2, end_dim=3)
        dim = self.token_len * 3

        self.att1 = Attention(dim, 4, dropout_rate)
        self.ff1 = FeedForward(dim, dropout_rate)

        self.att2 = Attention(dim, 4, dropout_rate)
        self.ff2 = FeedForward(dim, dropout_rate)

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim * self.num_tokens, 10)
        )

    def forward(self, x):
        x = self.patcher(x)
        x = x + self.positional_encoding
        x = self.flatten(x)
        res = self.att1(x)
        x = x + res
        res = self.ff1(x)
        x = x + res
        res = self.att2(x)
        x = x + res
        res = self.ff2(x)
        x = x + res
        x = self.out(x)
        return x
