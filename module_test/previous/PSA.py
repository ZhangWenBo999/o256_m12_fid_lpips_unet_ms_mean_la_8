import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = nn.Conv2d(dim, h, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.pe = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, -1, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
                (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W) + self.pe(v.reshape(B, -1, H, W))
        x = self.proj(x)
        return x


class PSA(nn.Module):

    def __init__(self, c1, c2, e=1):
        super().__init__()
        assert (c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(
            nn.Conv2d(self.c, self.c * 2, 1),
            nn.Conv2d(self.c * 2, self.c, 1, bias=False)
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


if __name__ == "__main__":
    input = torch.randn(1, 64, 128, 128)
    model = PSA(c1=64, c2=64)
    output = model(input)
    print("input:", input.shape)
    print("output:", output.shape)