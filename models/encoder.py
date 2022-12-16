import sys
import logging
import torch
import torch.nn as nn
from code_generation.exception import CodeGeneratorException
from models.multi_head_attention import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer
from typing import Optional
logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    def __init__(self, input_dim: Optional[int], hid_dim: Optional[int], enc_layers: Optional[int], enc_heads: Optional[int], enc_pf_dim:Optional[int], enc_dropout: Optional[float], device: Optional[str], max_length: Optional[int]):
        super().__init__()
        try:
            self.device = device
            self.tok_embedding = nn.Embedding(input_dim, hid_dim)
            self.pos_embedding = nn.Embedding(max_length, hid_dim)
            self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                    enc_heads, 
                                                    enc_pf_dim,
                                                    enc_dropout, 
                                                    device) 
                                        for _ in range(enc_layers)])
            self.dropout = nn.Dropout(enc_dropout)
            self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    def forward(self, src, src_mask):
        try:
            batch_size = src.shape[0]
            src_len = src.shape[1]
            pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
            for layer in self.layers:
                src = layer(src, src_mask)
                
            return src
        
        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim: int, enc_heads: int, enc_pf_dim: int, enc_dropout: float, device: str):
        super().__init__()
        try:
            self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
            self.ff_layer_norm = nn.LayerNorm(hid_dim)
            self.self_attention = MultiHeadAttentionLayer(hid_dim=hid_dim, n_heads=enc_heads, n_dropout=enc_dropout, device=device)
            self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim=hid_dim, 
                                                                        pf_dim=enc_pf_dim, 
                                                                        n_dropout=enc_dropout)
            self.dropout = nn.Dropout(enc_dropout)

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    def forward(self, src, src_mask):
        try:        
            #self attention
            _src, _ = self.self_attention(src, src, src, src_mask)
            
            #dropout, residual connection and layer norm
            src = self.self_attn_layer_norm(src + self.dropout(_src))
            
            #positionwise feedforward
            _src = self.positionwise_feedforward(src)
            
            #dropout, residual and layer norm
            src = self.ff_layer_norm(src + self.dropout(_src))
            
            return src

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e

        
