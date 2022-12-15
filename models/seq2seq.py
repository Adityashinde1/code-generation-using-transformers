import sys
import logging
from code_generation.exception import CodeGeneratorException
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device) -> None:
        super().__init__()
        try:
            self.encoder = encoder
            self.decoder = decoder
            self.src_pad_idx = src_pad_idx
            self.trg_pad_idx = trg_pad_idx
            self.device = device

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    def make_src_mask(self, src):
        try:
            #src = [batch size, src len]
            
            src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

            #src_mask = [batch size, 1, 1, src len]

            return src_mask

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e

    
    def make_trg_mask(self, trg):
        try:
            #trg = [batch size, trg len]
            
            trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
            
            #trg_pad_mask = [batch size, 1, 1, trg len]
            
            trg_len = trg.shape[1]
            
            trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
            
            #trg_sub_mask = [trg len, trg len]
                
            trg_mask = trg_pad_mask & trg_sub_mask
            
            #trg_mask = [batch size, 1, trg len, trg len]
            
            return trg_mask

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    def forward(self, src, trg):
        try:
            #src = [batch size, src len]
            #trg = [batch size, trg len]
                    
            src_mask = self.make_src_mask(src)
            trg_mask = self.make_trg_mask(trg)
            
            #src_mask = [batch size, 1, 1, src len]
            #trg_mask = [batch size, 1, trg len, trg len]
            
            enc_src = self.encoder(src, src_mask)
            
            #enc_src = [batch size, src len, hid dim]
                    
            output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
            
            #output = [batch size, trg len, output dim]
            #attention = [batch size, n heads, trg len, src len]
            
            return output, attention 

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e    