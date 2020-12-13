import math
import logging
import torch 
import torch.nn as nn
from torch.nn import functional as F 

logger = logging.getLogger()


class MultiheadedSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # projections
        self.n_embd = config.n_embd
        self.key = nn.Linear(self.n_embd,  self.n_embd)
        self.quary = nn.Linear( self.n_embd,  self.n_embd)
        self.value = nn.Linear( self.n_embd,  self.n_embd)
        # regularization by dropout
        self.attn_drop = nn.Dropout(config.attn_drop)
        self.resid_drop = nn.Dropout(config.resid_drop)
        # output projection??
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        
        if config.apply_mask == True:
            self.register_buffer('mask', )