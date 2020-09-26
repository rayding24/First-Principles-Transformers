import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable

'''
Ray Ding 
rayding2011@gmail.com
Sept. 2020
'''

def attention(Q, K, V):
    ''' Functional implementation for scaled dot product attention formula'''
    dot_prod = torch.matmul(Q, torch.transpose(K, -2, -1)) #swap last 2 dims, regardless of batch dim
    K_dim = K.size(-1)
    softmax = F.softmax(dot_prod/math.sqrt(K_dim), dim = -1)
    attention = torch.matmul(softmax, V)
    return attention



if __name__ == '__main__':
    Q = Variable(torch.rand(20, 50))
    K = Variable(torch.rand(20, 50))
    V = Variable(torch.rand(20, 1))
    a = attention(Q, K, V)
    print(a)