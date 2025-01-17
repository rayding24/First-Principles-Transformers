import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from torch import einsum

'''
Ray Ding 
rayding2011@gmail.com
Sept. 2020
'''

def attention(Q, K, V, mask=None): 
    ''' Functional implementation for scaled dot product attention formula with optional binary mask'''
    dot_prod = torch.matmul(Q, torch.transpose(K, -2, -1)) #swap last 2 dims, regardless of batch dim
    K_dim = K.size(-1)
    if mask:
        dot_prod[mask] = float('-inf') # proper mask for batch?
    softmax = F.softmax(dot_prod/math.sqrt(K_dim), dim = -1)
    attention = torch.matmul(softmax, V)
    return attention

        
class SelfAttentionWide(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        # we want to output the same dim as embedding to enable residual connection
        # init 3 matrices all the same
        self.M_Q, self.M_V, self.M_K = \
            [nn.Linear(emb_dim, emb_dim*num_heads, bias=False) for _ in range(3)]
        
        self.M_merge_heads = nn.Linear(emb_dim*num_heads, emb_dim, bias=False)
        
    def forward(self, x):
        # get Q, K, V
        Q = self.M_Q(x)
        K = self.M_K(x)
        V = self.M_V(x)
        multi_att = attention(Q, K, V)
        return self.M_merge_heads(multi_att)
        
        
if __name__ == '__main__':
    Q = Variable(torch.rand(20, 50))
    K = Variable(torch.rand(20, 50))
    V = Variable(torch.rand(20, 50))
    a = attention(Q, K, V)
    print(a)
    
    
    #experiment
