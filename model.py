import torch 
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model:int , vocab_size:int):
        super().__init__()
        self.d_model= d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)
        


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int , dropout: int)-> None:
        super().__init__()
        self.d_model= d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)
        
        # next we need to initialize a matrix of ( seq_len , d_model) aka seq_len tokens of d_model size 
        pe = torch.zeros(self.seq_len, self.d_model)
        
        # next we need to create the postions and denominator of following the formula on the paper , we use exp(log(formula)) because we can ensure calculation stability , we can't directly calculate it ( very big )
        positions=torch.arange(0,seq_len,dtype=float).unsqueeze(1)
        div_term= torch.exp(torch.arange(0,d_model,2).float*(-torch.log(10000)/d_model))
        # apply sin and cosin to even and odd positions of the vector respectively
        pe[:,0::2]=torch.sin(positions*div_term)
        pe[:,1::2]=torch.cos(positions*div_term)
        pe= pe.unsqueeze(0)
        # making the shape of the sentence (1 , seq , d_model ) since this will work onlty for 1 sentence but we have many sentences         
        self.register_buffer("pe",pe)
    def forward(self,x):
        x= x+self.pe[:,x.shape[1],:].requires_grad_(False) # Shape: (1, actual_seq_len, d_model)
        # Positional encodings are fixed and should not be learned during training
        return self.dropout(x)
    
    
class LayerNormalization(nn.Module):
    
    def __init__(self,eps:float=10**-6)->None:
        super().__init__()
        self.eps=eps # to ensure that we don't devide by zero 
        self.alpha=nn.Parameter(torch.ones(1))# it's the learnable parameter that we multiply the formula with
        self.bias=nn.Parameter(torch.zeros(1))# it's the learnable parameter that we add to the formula 
        
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)# dim= -1 meaning we want to calculate the mean of the last dimension of the tensor (d_model)
        std=x.std(dim=-1,keepdim=True) # and keepdim is used to make us be able to do the addition in the next step since we cant's do it without having same dimension z
        return self.alpha*(x-mean)/(std+self.eps) +self.bias 
    
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int , dropout:float):
        super().__init__()
        self.linear1= nn.Linear(d_model, d_ff) # from the paper it's W1 & B1
        self.dropout= nn.Dropout(dropout)
        self.linear2= nn.Linear(d_ff,d_model) # from the paper it's W2 & B2
    def forward(self , x): # FFN(x) = max(0, xW1 + b1)W2 + b2
        x= self.linear1(x)
        x = torch.relu(x)  # Apply ReLU 
        x= self.dropout(x)
        return self.linear2(x)
        