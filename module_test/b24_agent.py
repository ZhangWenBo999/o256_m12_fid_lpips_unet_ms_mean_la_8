import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

'''
Agent Attention: On the Integration of Softmax and Linear Attention (ECCV 2024)
即插即用模块：Agent Attention（代理注意力模块）- 替身模块
一、背景
Transformer中的注意力模块在计算机视觉任务中取得了显著的成功，但传统的Softmax全局注意力由于计算复杂度高
限制了其在高分辨率场景下的应用。而线性注意力虽然具有更低的计算复杂度，却在表达能力上有所欠缺。为了解决这一
难题，Agent-Attention通过引入“代理标记”（Agent Tokens）将Softmax和线性注意力结合起来，既保留了高效
的全局建模能力，又显著降低了计算复杂度，尤其适用于高分辨率视觉任务。

二、Agent-Attention模块原理
1. 整体结构
A. 输入特征：以查询（Q）、键（K）和值（V）为核心构成传统注意力机制。
B. 代理标记（Agent Tokens, A）：通过池化或学习的方式生成，代理原始查询标记以聚合全局信息。
C. 两阶段操作：
   a. Agent Aggregation（信息聚合）：代理标记从键值对中提取全局信息。
   b. Agent Broadcast（信息分发）：代理标记将信息广播回查询标记。
D. 输出特征：经过增强后的特征矩阵。
2. 关键模块
A. Softmax Attention：对代理标记与原始标记进行两次Softmax操作，实现信息的聚合与分发。
B. Agent Bias：引入位置编码以增强空间信息。
C. 深度卷积（DWC）：恢复特征多样性以避免信息损失。
3. 计算复杂度
Quadratic → Linear：通过代理标记减少全局交互次数，将传统的O(N²)复杂度降至O(Nn)，其中n
为代理标记数，N为输入特征数。

三、适用任务
适用于图像分类、目标检测、语义分割图像生成等所有计算机视觉任务。
'''

class AgentAttention(nn.Module):
    # def __init__(self, dim, num_patches=1024, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
    #              sr_ratio=1, agent_num=49, **kwargs):
    def __init__(self, dim, num_patches=4096, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    # def forward(self, x, H=32, W=32):
    def forward(self, x, H=64, W=64):

        b1,c1,h1,w1 = x.shape
        x = x.reshape(b1,c1,-1).transpose(-1,-2)
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(-1,-2).reshape(b1,c1,h1,w1)
        return x
if __name__ == '__main__':
    dim = 128  #通道数
    num_patches = 256 #序列长度
    block = AgentAttention(dim=dim)
    H, W = 16, 16
    x = torch.randn(1,dim, H,W)
    output = block(x)
    print(f"Input size: {x.size()}")
    print(f"Output size: {output.size()}")