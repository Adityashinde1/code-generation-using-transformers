import sys
import logging
import torch
import torch.nn as nn
from code_generation.exception import CodeGeneratorException
from models.multi_head_attention import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer

logger = logging.getLogger(__name__)

class Decoder(nn.Module):
    def __init__(self, output_dim: int, hid_dim: int, dec_layers: int, dec_heads: int, dec_pf_dim: int, dec_dropout: float, device: str, max_length: int) -> None:
        super().__init__()
        try:
            self.device = device
            self.tok_embedding = nn.Embedding(output_dim, hid_dim)
            self.pos_embedding = nn.Embedding(max_length, hid_dim)
            self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                    dec_heads, 
                                                    dec_pf_dim, 
                                                    dec_dropout, 
                                                    device)
                                        for _ in range(dec_layers)])
            self.fc_out = nn.Linear(hid_dim, output_dim)
            self.dropout = nn.Dropout(dec_dropout)
            self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e

        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        try:   
            batch_size = trg.shape[0]
            trg_len = trg.shape[1]
            pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
            for layer in self.layers:
                trg, attention = layer(trg, enc_src, trg_mask, src_mask)
            output = self.fc_out(trg)
        
            return output, attention  

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e      


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim: int, dec_heads: int, dec_pf_dim: int, dec_dropout: float, device: str) -> None:
        super().__init__()
        try:
            self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
            self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
            self.ff_layer_norm = nn.LayerNorm(hid_dim)
            self.self_attention = MultiHeadAttentionLayer(hid_dim=hid_dim, n_heads=dec_heads, n_dropout=dec_dropout, device=device)
            self.encoder_attention = MultiHeadAttentionLayer(hid_dim=hid_dim, n_heads=dec_heads, n_dropout=dec_dropout, device=device)
            self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim=hid_dim, 
                                                                        pf_dim=dec_pf_dim, 
                                                                        n_dropout=dec_dropout)
            self.dropout = nn.Dropout(dec_dropout)

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e 

        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        try:
            _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
            
            #dropout, residual connection and layer norm
            trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
                
            #encoder attention
            _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
            # query, key, value
    
            #dropout, residual connection and layer norm
            trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

            #positionwise feedforward
            _trg = self.positionwise_feedforward(trg)
            
            #dropout, residual and layer norm
            trg = self.ff_layer_norm(trg + self.dropout(_trg))
            
            #trg = [batch size, trg len, hid dim]
            #attention = [batch size, n heads, trg len, src len]
            
            return trg, attention

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e
