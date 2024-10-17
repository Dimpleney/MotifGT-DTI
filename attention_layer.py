import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def scaled_dot_product(q, k, v, mask=None):
    
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)

        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        # print("qkv after permute:", qkv.shape)
        q, k, v = qkv.chunk(3, dim=-1)
        # print("q:", q.shape)
        # print("k:", k.shape)
        # print("v:", v.shape)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        # print("values:",values.shape)
        # print("attention:",attention.shape)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        # print("values after permute:",values.shape)
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        # print("values after reshape:",values.shape)

        o = self.o_proj(values)
        # print("o:",o.shape)
        if return_attention:
            return o, attention
        else:
            return o
