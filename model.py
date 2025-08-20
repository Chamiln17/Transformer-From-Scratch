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
        # (Batch , seq_len , d_model ) --> (Batch , seq_len , d_ff )--> ( batch , seq_len , d_model) 
        x= self.linear1(x)
        x = torch.relu(x)  # Apply ReLU 
        x= self.dropout(x)
        return self.linear2(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int , h:int , dropout: float):
        super().__init__()
        self.d_model= d_model
        self.h=h
        assert d_model % h == 0 , "d_model is not divisible by h"
        
        self.d_k = d_model // h
        self.Wq= nn.Linear(d_model,d_model)
        self.Wk= nn.Linear(d_model,d_model)
        self.Wv= nn.Linear(d_model,d_model)

        self.dropout=nn.Dropout(dropout)
        self.Wo=nn.Linear(d_model,d_model)
        
    @staticmethod
    def attention(query , key , value , mask , dropout: nn.Dropout):
        d_k= query.shape[-1]
        
        #(batch, h, seq_len , d_k ) --> (batch , h , seq_len , seq_len) (self attention matrix kinda)
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        
        if mask is not None : 
            attention_scores.masked_fill_(mask==0, -1e-9)
        attention_scores= attention_scores.softmax(dim=-1)# (batch , h , seq_len , seq_len)
        if dropout is not None: 
            attention_scores= dropout(attention_scores) 
        return (attention_scores @ value) , attention_scores # for visualizations we return the attention scores too , the final dim is (batch ,h , seq_len , d_k)
        
    def forward(self , q , k , v , mask): 
        query = self.Wq(q) # (batch , seq_len , d_model)
        key= self.Wk(k) # (batch , seq_len , d_model)
        value= self.Wv(v) # (batch , seq_len , d_model)
        
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # (batch, seq_len , h, k) --> (batch, h , seq_len , k) so we want that each head will have a view for all the sentence but only K part of the embbedings
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) # (batch, seq_len , h, k) --> (batch, h , seq_len , k) so we want that each head will have a view for all the sentence but only K part of the embbedings
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) # (batch, seq_len , h, k) --> (batch, h , seq_len , k) so we want that each head will have a view for all the sentence but only K part of the embbedings
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query , key , value , mask , self.dropout)  #(batch ,h , seq_len , d_k)
        
        # (batch ,h , seq_len , d_k) --> (batch , seq_len , h, d_k) --> (batch , seq_len , d_model) final goal
        x= x.transpose(1,2).contiguous().view(x.shape[0],-1, self.h*self.d_k) # we use contiguous since After transpose(), tensor data might not be stored contiguously in memory. contiguous() ensures the tensor is stored in a contiguous block, which is required for view() operations. 
        
        # we started from (batch , seq_len , d_model) to (batch , seq_len , d_model)
        return self.Wo(x)
        
class ResidualConnection(nn.Module):
    def __init__(self, dropout:float):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()
        
    def forward(self, x , sublayer):
        return x+ self.dropout(sublayer(self.norm(x))) # in paper it is self.norm(sublayer(x)) but many implmentations do this 
        
        
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,feedforward_block:FeedForward,dropout:float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feedforward_block=feedforward_block
        self.residual_connection= nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x= self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x= self.residual_connection[1](x, lambda x: self.feedforward_block(x))
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers =layers
        self.norm=LayerNormalization()
        
    def forward(self,x, mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)
        
class DecoderBlock(nn.Module):
    def __init__(self ,self_attention_block:MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForward , dropout: float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)]) 
    def forward(self , x ,encoder_output, enc_mask, dec_mask):
        x=self.residual_connections[0](x,lambda x: self.self_attention_block(x,x,x,dec_mask))
        x=self.residual_connections[1](x,lambda x: self.cross_attention_block(x,encoder_output,encoder_output,enc_mask))
        x=self.residual_connections[2](x,lambda x: self.feed_forward_block(x))
        return x
        
class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()
    def forward(self, x , encoder_output, enc_mask , dec_mask):
        for layer in self.layers:
            x=layer(x, encoder_output,enc_mask,dec_mask) # same as in the encoder and the encoder output is shared across all the decoder blocks
        return self.norm(x)
        



         