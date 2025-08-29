import torch 
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src , tokenizer_tgt , src_lang , tgt_lang, seq_len):
        super().__init__()
        self.ds=ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang
        self.seq_len=seq_len
    
    
        self.sos_token=torch.Tensor([tokenizer_src.token_to_id(["[SOS]"])], dtype=torch.int64)
        self.eos_token=torch.Tensor([tokenizer_src.token_to_id(["[EOS]"])], dtype=torch.int64)
        self.pad_token=torch.Tensor([tokenizer_src.token_to_id(["[PAD]"])], dtype=torch.int64)
    def __len__(self):
        return len(self.ds)
    

    def __getitem__(self, index):
        src_target_pair= self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text= src_target_pair["translation"][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) -2 # eos and sos tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) -1 # only eos token
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("sentence is too long")
        
        # adding SOS & EOS token and the padding at the end 

        encoder_inputs = torch.cat(
            
            [
                self.sos_token,
                torch.Tensor(enc_input_tokens),
                self.eos_token,
                torch.Tensor(enc_num_padding_tokens * [self.pad_token], dtype=torch.int64)
                
            ]
        )
        
        # adding only SOS special token
        decoder_inputs = torch.cat(
            
            [
                self.sos_token,
                torch.Tensor(enc_input_tokens, dtype=torch.int64 ),
                torch.Tensor(dec_num_padding_tokens * [self.pad_token], dtype=torch.int64)
                
            ]
        )
        # adding only EOS special token

        label = torch.cat(
            
            [
                torch.Tensor(enc_input_tokens, dtype=torch.int64 ),
                self.eos_token,
                torch.Tensor(dec_num_padding_tokens * [self.pad_token], dtype=torch.int64)
                
            ]
        )
        assert encoder_inputs.size(0) == self.seq_len
        assert decoder_inputs.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_inputs":encoder_inputs,# [seq_len]
            "decoder_inputs":decoder_inputs, # [seq_len]
            "encoder_mask":(encoder_inputs != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # [1, 1, seq_len] 
            "decoder_mask":(decoder_inputs != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_inputs.size(0)), # [1, seq_len, seq_len] 
            "label":label ,# [seq_len]
            "src_text":src_text,
            "tgt_text":tgt_text,
        }
        
def causal_mask(self, size):
    mask = torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask ==0
            
        



            
            
    
        
    