# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:57:19 2023

@author: Nithesh
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(1337)

# %% Reading Input data
filename = 'input.txt'
with open(filename, 'r') as f:
    text = f.read()
f.close()

# %%  Tokenization
vocab = sorted(list(set(text)))
word2idx = {vocab[i]: i for i in range(len(vocab))}
idx2word = {i: vocab[i] for i in range(len(vocab))}
def encode(s): return [word2idx[c] for c in s]
def decode(i): return ''.join([idx2word[idx] for idx in i])


data = torch.tensor(encode(text), dtype=torch.long)
# %%
split = int(0.9*len(data))
trainData = data[:split]
testData = data[split:]

block_size = 8
batch_size = 4
eval_iters = 200

# %% Batch Generator
def createBatch(split):
    data = trainData if split == 'train' else testData
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

#%%
@torch.no_grad()
def estimateLoss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = createBatch(split)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


x, y = createBatch('train')

# for b in range(batch_size):
#     for t in range(block_size):
#         context = x[b, :t+1]
#         target = y[b, t]
#         print(f"When input is {context.tolist()}, the target is {target}")

#%% Bigram Model
class BiGramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.batch_size = 32

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is of shape (B,T)
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # becomes (B,C) as only last T is selected
            logits = logits[:,-1, :]
            
            # converting logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sampling from the prob distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # append smapled index to the running sequence (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    # def backprop(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr = 1e-3)
    #     batch

#%% SelfAttentionModel
class SelfAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_eDim)
        self.positional_encoding_table = nn.Embedding(block_size,n_eDim)
        # self.sa_heads = MultiHeadAttention(4, n_eDim//4)
        self.blocks = nn.Sequential(
            Block(n_eDim, num_heads = 6),
            Block(n_eDim, num_heads = 6),
            Block(n_eDim, num_heads = 6),
            Block(n_eDim, num_heads = 6),
            Block(n_eDim, num_heads = 6),
            Block(n_eDim, num_heads = 6),
            nn.LayerNorm((n_eDim))
            )
        self.lm_head = nn.Linear(n_eDim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embds = self.token_embedding_table(idx) #(B,T,C)
        pos_embds = self.positional_encoding_table(torch.arange(T)) #(T,C)
        x = token_embds + pos_embds #(B,T,C)
        x = self.blocks(x) #(B,T,C)
        # x = self.ffw(x) #(B,T,C)
        logits = self.lm_head(x) #(B,T, vocab_size)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is of shape (B,T)
        for _ in range(max_new_tokens):
            idx_crop = idx[:,- block_size:]
            logits, loss = self(idx_crop)
            # becomes (B,C) as only last T is selected
            logits = logits[:,-1, :]
            
            # converting logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sampling from the prob distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # append smapled index to the running sequence (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
 
#%% Self Attention Head
class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_eDim, head_size, bias = False)
        self.query = nn.Linear(n_eDim, head_size, bias = False)
        self.value = nn.Linear(n_eDim, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T, head_size
        q = self.query(x) # (B,T, head_size)
        wei = q @ k.transpose(-2,-1)* C**-0.5 #(B,T,C) @ (B,C,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #(B,T,T)
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

#%% Feed Forward Layer
class FeedForward(nn.Module):
    def __init__(self, n_eDim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_eDim, 4*n_eDim),
            nn.ReLU(),
            nn.Linear(4*n_eDim, n_eDim),
            nn.Dropout(dropout))
        
    def forward(self, x):
        return self.net(x)
#%% Multi-Head Attention Model
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.mheads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_eDim, n_eDim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        out = torch.cat([h(x) for h in self.mheads], dim = -1) 
        out = self.proj(out)
        out = self.dropout(out)
        return  out
#%% Decoder block    
class Block(nn.Module):
    def __init__(self, n_eDim, num_heads):
        super().__init__()
        head_size = n_eDim//num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffw = FeedForward(n_eDim)
        self.lnorm1 = nn.LayerNorm(n_eDim)
        self.lnorm2 = nn.LayerNorm(n_eDim)
        
        
    def forward(self, x):
        x = x + self.sa(self.lnorm1(x))
        x = x + self.ffw(self.lnorm2(x))
        return x
   
#%% Layer/Batch Normalization
class BatchNorm1D:
    def __init__(self, dim, eps = 1e-5, momentum = 0.1):
        self.eps = eps        
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self,x):
        xmean = x.mean(1, keepDim = True) #batch mean
        xvar = x.var(1, keepDim = True)
        xhat = (x-xmean)/torch.sqrt(xvar + self.eps)
        self.out = self.gamma*xhat  + self.beta
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
#%%
block_size = 64
batch_size = 256
max_iter_num = 5000
iter_interval = 500
eval_iters = 200
n_eDim = 384
num_heads = 6
num_layers = 6
vocab_size = len(vocab)
lr = 3e-4
dropout = 0.2


m = SelfAttentionModel()
logits, loss = m(x, y)

optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
for iter in range(max_iter_num):
    
    # Verbose
    if iter % iter_interval == 0:
        losses = estimateLoss(m)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    #create Batches
    xb, yb = createBatch('train')
    
    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#%% Generate from the Model
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context,max_new_tokens=1000)[0].tolist()))