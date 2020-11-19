import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from torch import einsum



class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.attention_layer = SelfAttentionWide(emb_dim, num_heads)
        self.layernorm1 = nn.LayerNorm(emb_dim)
        self.layernorm2 = nn.LayerNorm(emb_dim)
        # this can be pretty much any mlp we want
        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU())
        
    def forward(self, x):
        x = x + self.attention_layer(x)
        x = self.layernorm1(x)
        x = x + self.mlp(x)
        return self.layernorm2(x)