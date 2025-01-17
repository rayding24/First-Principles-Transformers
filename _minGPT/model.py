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
        self.n_head = config.n_head
        # TODO: bias should be false here, does it make a diff. in performance though?
        self.key = nn.Linear(self.n_embd,  self.n_embd)
        self.query = nn.Linear( self.n_embd,  self.n_embd)
        self.value = nn.Linear( self.n_embd,  self.n_embd)
        # regularization by dropout
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # why need output projection??
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        
        self.block_size = config.block_size
        if config.apply_mask == True:
            self.register_buffer('mask', torch.tril(torch.ones(self.block_size, self.block_size))
                                 .view(1,1, self.block_size, self.block_size) ) #expand dims
        
    def forward(self, x):
        B,T,C = x.size() # batch size, bptt size?, chunk size to be divided by heads?
        
        k = self.key(x).view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # // here is probably for numerical stability?
        q = self.query(x).view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        # why move head forward to be the batch dim?? 
        
        # now q, k, v are just tensors and @ can be used
        
        att = q@ (k.transpose(-2, -1)) # vector math to batches becomes this mess
        att /= math.sqrt(k.size(-1)) # so variance stays constant as dim scales
        
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf')) #TODO: this is weird, replace with att[mask] = -inf
        att = F.softmax(att, dim=-1) # not intuitive but from vector math is
        att = self.attn_drop(att)
        y = att @ v 
        y = y.transpose(1,2).contiguous().view(B,T,C)
        
        y = self.resid_drop(self.proj(y)) # what is this for?
        return y
    
    
    
class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadedSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 5*config.n_embd),
            nn.GELU(),
            nn.Linear(5*config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )
        
    def forward(self, x):
        '''
        The original minGPT implementation seems to be somewhat off
        compared to the Decoder Archetecture from the GPT paper
        So here is my version:
        '''
        x = x + self.attn(x)
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x 
    
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        
        
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer) ])
        
        # head
        self.ln = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, 10, bias=False)
        self.block_size = config.block_size

        self.apply(self._init_weights)
        
        
    def _init_weights(self, module):
        ''' 
        Not really necessary when Kaiming He does it for you :P but good practice nonetheless
        '''
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
            
    def forward(self, idx, targets=None):
        _, t = idx.size() # index for words
        
        # assert t <= self.block_size, 'cannot forward, model block size is exhausted, wtf does this mean???'
        
        token_embeddings = self.tok_emb(idx) # index to vec
        positional_embeddings = self.pos_emb[:, :t, :] #wtf is this?? pos emb straight from Parameter??
        x = token_embeddings+positional_embeddings
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln(x)
        x = torch.mean(x, dim=1)
        logits = self.head(x)
            
        return logits 
        
        
class GPTConfig:
    
    def __init__(self, vocab_size, block_size, embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1,
    max_grad_clip_norm=100, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop 
        self.max_grad_clip_norm = max_grad_clip_norm

        for k,v in kwargs.items():
            setattr(self, k, v)
            
            

            