import os
import sys
import io
import random
import keyword
from code_generation.exception import CodeGeneratorException
from code_generation.entity.artifacts_entity import DataTransformationArtifacts
from code_generation.entity.config_entity import ModelTrainerAndEvalConfig
from code_generation.entity.artifacts_entity import ModelTrainerAndEvalArtifacts
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from code_generation.constants import *
from code_generation.utils.main_utils import MainUtils
from models.loss_function import CrossEntropyLoss
from code_generation.configuration.gcloud import GCloud
import logging
import spacy
from tokenize import tokenize
import time
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from torchtext.legacy.data import Field, BucketIterator, Iterator

SEED = 1235
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy.load('en_core_web_sm')

logger = logging.getLogger(__name__)


class ModelTrainerAndEval:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts, model_trainer_and_eval_config: ModelTrainerAndEvalConfig, 
                    utils: MainUtils, gcloud=GCloud):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_and_eval_config = model_trainer_and_eval_config
        self.utils = utils
        self.gcloud = gcloud


    @staticmethod
    def initialize_weights(m):
        try:
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    @staticmethod
    def maskNLLLoss(inp, target, mask, trg_pad_idx, device):
        try:
            # print(inp.shape, target.shape, mask.sum())
            nTotal = mask.sum()
            crossEntropy = CrossEntropyLoss(ignore_index = trg_pad_idx, smooth_eps=0.20)
            loss = crossEntropy(inp, target)
            loss = loss.to(device)
            return loss, nTotal.item()

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e

        
    @ staticmethod
    def make_trg_mask(trg, trg_pad_idx, device):
        try:
            #trg = [batch size, trg len]
            
            trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)
            
            #trg_pad_mask = [batch size, 1, 1, trg len]
            
            trg_len = trg.shape[1]
            
            trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = device)).bool()
            
            #trg_sub_mask = [trg len, trg len]
                
            trg_mask = trg_pad_mask & trg_sub_mask
            
            #trg_mask = [batch size, 1, trg len, trg len]
            
            return trg_mask

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    def train(self, model, iterator, optimizer, criterion, clip, trg_pad_idx, device):
        try:
            model.train()
            
            n_totals = 0
            print_losses = []
            for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
                # print(batch)
                loss = 0
                src = batch.Input.permute(1, 0)
                trg = batch.Output.permute(1, 0)
                trg_mask = self.make_trg_mask(trg, trg_pad_idx, device)
                optimizer.zero_grad()
                
                output, _ = model(src, trg[:,:-1])
                        
                #output = [batch size, trg len - 1, output dim]
                #trg = [batch size, trg len]
                    
                output_dim = output.shape[-1]
                    
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)
                        
                #output = [batch size * trg len - 1, output dim]
                #trg = [batch size * trg len - 1]
                    
                mask_loss, nTotal = criterion(output, trg, trg_mask, trg_pad_idx, device)
                
                mask_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                
                optimizer.step()
                
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

            return sum(print_losses) / n_totals

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    def evaluate(self, model, iterator, criterion, trg_pad_idx, device):
        try:
            model.eval()
            
            n_totals = 0
            print_losses = []
            
            with torch.no_grad():
            
                for i, batch in tqdm(enumerate(iterator), total=len(iterator)):

                    src = batch.Input.permute(1, 0)
                    trg = batch.Output.permute(1, 0)
                    trg_mask = self.make_trg_mask(trg, trg_pad_idx, device)

                    output, _ = model(src, trg[:,:-1])
                    
                    #output = [batch size, trg len - 1, output dim]
                    #trg = [batch size, trg len]
                    
                    output_dim = output.shape[-1]
                    
                    output = output.contiguous().view(-1, output_dim)
                    trg = trg[:,1:].contiguous().view(-1)
                    
                    #output = [batch size * trg len - 1, output dim]
                    #trg = [batch size * trg len - 1]
                    
                    mask_loss, nTotal = criterion(output, trg, trg_mask, trg_pad_idx, device)

                    print_losses.append(mask_loss.item() * nTotal)
                    n_totals += nTotal

            return sum(print_losses) / n_totals

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    @staticmethod
    def epoch_time(start_time, end_time):
        try:
            elapsed_time = end_time - start_time
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
            return elapsed_mins, elapsed_secs

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    @staticmethod
    def tokenize_python_code(python_code_str, mask_factor=0.3) -> list:

        try: 
            var_dict = {} # Dictionary that stores masked variables

            # certain reserved words that should not be treated as normal variables and
            # hence need to be skipped from our variable mask augmentations
            skip_list = ['range', 'enumerate', 'print', 'ord', 'int', 'float', 'zip'
                        'char', 'list', 'dict', 'tuple', 'set', 'len', 'sum', 'min', 'max']
            skip_list.extend(keyword.kwlist)

            var_counter = 1
            python_tokens = list(tokenize(io.BytesIO(python_code_str.encode('utf-8')).readline))
            tokenized_output = []

            for i in range(0, len(python_tokens)):
                if python_tokens[i].type == 1 and python_tokens[i].string not in skip_list:
                
                    if i>0 and python_tokens[i-1].string in ['def', '.', 'import', 'raise', 'except', 'class']: # avoid masking modules, functions and error literals
                        skip_list.append(python_tokens[i].string)
                        tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
                    elif python_tokens[i].string in var_dict:  # if variable is already masked
                        tokenized_output.append((python_tokens[i].type, var_dict[python_tokens[i].string]))
                    elif random.uniform(0, 1) > 1-mask_factor: # randomly mask variables
                        var_dict[python_tokens[i].string] = 'var_' + str(var_counter)
                        var_counter+=1
                        tokenized_output.append((python_tokens[i].type, var_dict[python_tokens[i].string]))
                    else:
                        skip_list.append(python_tokens[i].string)
                        tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
            
                else:
                    tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
            
            return tokenized_output

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    @staticmethod
    def create_data_example(train_df: DataFrame, val_df: DataFrame, train_expansion_factor: int, fields: list) -> list:
        try:
            train_example = []
            val_example = []

            for j in range(train_expansion_factor):
                for i in range(train_df.shape[0]):
                    try:
                        ex = data.Example.fromlist([train_df.question[i], train_df.solution[i]], fields)
                        train_example.append(ex)
                    except:
                        pass
            
            for i in range(val_df.shape[0]):
                try:
                    ex = data.Example.fromlist([val_df.question[i], val_df.solution[i]], fields)
                    val_example.append(ex)
                except:
                    pass

            return train_example, val_example

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    def initiate_model_and_eval_trainer(self) -> ModelTrainerAndEvalArtifacts:
        try:
            os.makedirs(self.model_trainer_and_eval_config.model_trainer_and_eval_artifacts_dir, exist_ok=True)
            logger.info(
                f"Created {os.path.basename(self.model_trainer_and_eval_config.model_trainer_and_eval_artifacts_dir)} directory."
            ) 

            train_df = self.utils.load_pickle_file(filepath=self.data_transformation_artifacts.train_df_path)
            val_df = self.utils.load_pickle_file(filepath=self.data_transformation_artifacts.test_df_path)

            Input = data.Field(tokenize = 'spacy', init_token='', eos_token='', lower=True)
            Output = data.Field(tokenize = self.tokenize_python_code, init_token='', eos_token='', lower=False)

            fields = [('Input', Input),('Output', Output)]

            train_example, val_example = self.create_data_example(train_df=train_df, val_df=val_df, train_expansion_factor=TRAIN_EXPANSION_FACTOR, fields=fields)

            train_data = data.Dataset(train_example, fields)
            valid_data =  data.Dataset(val_example, fields)

            Input.build_vocab(train_data, min_freq = 0)
            Output.build_vocab(train_data, min_freq = 0)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            input_dim = len(Input.vocab)
            output_dim = len(Output.vocab)
            hid_dim = HID_DIM
            enc_layer = ENC_LAYERS
            dec_layer = DEC_LAYERS
            enc_head = ENC_HEADS
            dec_head = DEC_HEADS
            enc_pf_dim = ENC_PF_DIM
            dec_pf_dim = DEC_PF_DIM
            enc_dropout = ENC_DROPOUT
            dec_dropout = DEC_DROPOUT

            encoder = Encoder(input_dim=input_dim, hid_dim=hid_dim, enc_layers=enc_layer, enc_heads=enc_head, 
                            enc_pf_dim=enc_pf_dim, enc_dropout=enc_dropout, device=device, max_length=ENC_MAX_LENGTH)

            decoder = Decoder(output_dim=output_dim, hid_dim=hid_dim, dec_layers=dec_layer, dec_heads=dec_head,
                             dec_pf_dim=dec_pf_dim, dec_dropout=dec_dropout, device=device, max_length=DEC_MAX_LENGTH)

            src_pad_idx = Input.vocab.stoi[Input.pad_token]
            trg_pad_idx = Output.vocab.stoi[Output.pad_token]

            model = Seq2Seq(encoder=encoder, decoder=decoder, src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, device=device)

            model.apply(self.initialize_weights)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = self.maskNLLLoss

            best_valid_loss = BEST_VALID_LOSS

            for epoch in range(EPOCHS):
                print(f"epoch = {epoch+1}")
                start_time = time.time()
                
                train_example = []
                val_example = []

                for i in range(train_df.shape[0]):
                    try:
                        ex = data.Example.fromlist([train_df.question[i], train_df.solution[i]], fields)
                        train_example.append(ex)
                    except:
                        pass

                for i in range(val_df.shape[0]):
                    try:
                        ex = data.Example.fromlist([val_df.question[i], val_df.solution[i]], fields)
                        val_example.append(ex)
                    except:
                        pass       

                train_data = data.Dataset(train_example, fields)
                valid_data =  data.Dataset(val_example, fields)

                train_iterator, valid_iterator = BucketIterator.splits((train_data, valid_data), batch_size = BATCH_SIZE, 
                                                                            sort_key = lambda x: len(x.Input),
                                                                            sort_within_batch=True, device = device)                  

                train_loss = self.train(model, train_iterator, optimizer, criterion, CLIP, trg_pad_idx, device)
                valid_loss = self.evaluate(model, valid_iterator, criterion, trg_pad_idx, device)
                
                end_time = time.time()
                
                epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), self.model_trainer_and_eval_config.model_path)
                
                print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


            model_trainer_artifacts = ModelTrainerAndEvalArtifacts(trained_model_path=self.model_trainer_and_eval_config.model_upload_path)

            return model_trainer_artifacts

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e
