import os
import sys
import io
import random
import keyword
import time
import logging
import spacy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
from pandas import DataFrame
from datetime import datetime
from tokenize import tokenize
from torchtext.legacy import data
from torchtext.legacy.data import BucketIterator
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

SEED = 1235
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Loading the spacy english model
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
    def initialize_weights(m) -> None:
        logger.info("Enetered the initialize_weights method of Model trainer and eval class")
        try:
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    @staticmethod
    def maskNLLLoss(inp, target, mask, trg_pad_idx, device) -> float:
        logger.info("Enetered the maskNLLLoss method of Model trainer and eval class")
        try:
            nTotal = mask.sum()
            crossEntropy = CrossEntropyLoss(ignore_index = trg_pad_idx, smooth_eps=0.20)
            loss = crossEntropy(inp, target)
            loss = loss.to(device)

            logger.info("Exited the maskNLLLoss method of Model trainer and eval class")
            return loss, nTotal.item()

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e

        
    @ staticmethod
    def make_trg_mask(trg, trg_pad_idx, device) -> Tensor:
        logger.info("Enetered the make_trg_mask method of Model trainer and eval class")
        try:
            trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)
            trg_len = trg.shape[1]
            trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = device)).bool()  
            trg_mask = trg_pad_mask & trg_sub_mask
            
            logger.info("Exited the make_trg_mask method of Model trainer and eval class")
            return trg_mask

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    def train(self, model, iterator, optimizer, criterion, clip, trg_pad_idx, device) -> float:
        logger.info("Entered the train method of Model trainer and eval class")
        try:
            model.train()
            n_totals = 0
            print_losses = []
            for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
                loss = 0
                src = batch.Input.permute(1, 0)
                trg = batch.Output.permute(1, 0)
                trg_mask = self.make_trg_mask(trg, trg_pad_idx, device)
                optimizer.zero_grad()
                
                output, _ = model(src, trg[:,:-1])
                output_dim = output.shape[-1]
                    
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)
                    
                mask_loss, nTotal = criterion(output, trg, trg_mask, trg_pad_idx, device)
                mask_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

            logger.info("Exited the train method of Model trainer and eval class")
            return sum(print_losses) / n_totals

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    def evaluate(self, model, iterator, criterion, trg_pad_idx, device) -> float:
        logger.info("Entered the evaluate method of Model trainer and eval class")
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
                    output_dim = output.shape[-1]
                    
                    output = output.contiguous().view(-1, output_dim)
                    trg = trg[:,1:].contiguous().view(-1)

                    mask_loss, nTotal = criterion(output, trg, trg_mask, trg_pad_idx, device)
                    print_losses.append(mask_loss.item() * nTotal)
                    n_totals += nTotal

            logger.info("Exited the evaluate method of Model trainer and eval class")
            return sum(print_losses) / n_totals

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    @staticmethod
    def epoch_time(start_time: datetime, end_time: datetime) -> int:
        logger.info("Entered the epoch_time method of Model trainer and eval class")
        try:
            elapsed_time = end_time - start_time
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

            logger.info("Exited the epoch_time method of Model trainer and eval class")
            return elapsed_mins, elapsed_secs

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    @staticmethod
    def tokenize_python_code(python_code_str: str, mask_factor: float=0.3) -> list:
        logger.info("Entered the tokenize_python_code method of Model trainer and eval class")
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

            logger.info("Exited the tokenize_python_code method of Model trainer and eval class")
            return tokenized_output

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    @staticmethod
    def create_data_example(train_df: DataFrame, val_df: DataFrame, train_expansion_factor: int, fields: list) -> list:
        logger.info("Entered the create_data_example method of Model trainer and eval class")
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

            logger.info("Exited the create_data_example method of Model trainer and eval class")
            return train_example, val_example

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    def initiate_model_and_eval_trainer(self) -> ModelTrainerAndEvalArtifacts:
        logger.info("Entered the initiate_model_and_eval_trainer method of Model trainer and eval class")
        try:
            os.makedirs(self.model_trainer_and_eval_config.model_trainer_and_eval_artifacts_dir, exist_ok=True)
            logger.info(
                f"Created {os.path.basename(self.model_trainer_and_eval_config.model_trainer_and_eval_artifacts_dir)} directory."
            ) 
            # Loading train and test pickle file from artifacts directory 
            train_df = self.utils.load_pickle_file(filepath=self.data_transformation_artifacts.train_df_path)
            logger.info(f"Loaded {os.path.basename(self.data_transformation_artifacts.train_df_path)} file in model trainer and eval component.")
            val_df = self.utils.load_pickle_file(filepath=self.data_transformation_artifacts.test_df_path)
            logger.info(f"Loaded {os.path.basename(self.data_transformation_artifacts.test_df_path)} file in model trainer and eval component.")

            Input = data.Field(tokenize = 'spacy', init_token='', eos_token='', lower=True)
            Output = data.Field(tokenize = self.tokenize_python_code, init_token='', eos_token='', lower=False)

            fields = [('Input', Input),('Output', Output)]

            # Creating the train and test example 
            train_example, val_example = self.create_data_example(train_df=train_df, val_df=val_df, train_expansion_factor=TRAIN_EXPANSION_FACTOR, fields=fields)
            logger.info("Created Train and test examples.")

            train_data = data.Dataset(train_example, fields)
            valid_data =  data.Dataset(val_example, fields)

            Input.build_vocab(train_data, min_freq = 0)
            Output.build_vocab(train_data, min_freq = 0)
            logger.info("Vocab built")

            # saving the vocab
            self.utils.dump_pickle_file(output_filepath=self.model_trainer_and_eval_config.source_vocab_file_path, data=Input)
            logger.info(f"Source vocab file saved in the artifacts directory. File name - {os.path.basename(self.model_trainer_and_eval_config.source_vocab_file_path)}")
            self.utils.dump_pickle_file(output_filepath=self.model_trainer_and_eval_config.target_vocab_file_path, data=Output)
            logger.info(f"Target vocab file saved in the artifacts directory. File name - {os.path.basename(self.model_trainer_and_eval_config.target_vocab_file_path)}")

            # Uploading the vocab to cloud for inferencing
            self.gcloud.sync_folder_to_gcloud(gcp_bucket_url=BUCKET_NAME, filepath=self.model_trainer_and_eval_config.source_file_to_gcp_path, filename=SOURCE_VOCAB_FILE_NAME)
            logger.info(f"source vocab file uploaded to google container registry. File name - {SOURCE_VOCAB_FILE_NAME}")
            self.gcloud.sync_folder_to_gcloud(gcp_bucket_url=BUCKET_NAME, filepath=self.model_trainer_and_eval_config.target_file_to_gcp_path, filename=TARGET_VOCAB_FILE_NAME)
            logger.info(f"Target vocab file uploaded to google container registry. File name - {TARGET_VOCAB_FILE_NAME}")
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

            # Creating model instance
            model = Seq2Seq(encoder=encoder, decoder=decoder, src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, device=device)

            # Saving object of Seq2seq model instance
            torch.save(model, self.model_trainer_and_eval_config.seq_2_seq_model_instance_path)
            logger.info(f"Saved Seq2seq model instance to artifacts directory. File name - {self.model_trainer_and_eval_config.seq_2_seq_model_instance_path}")

            # Uploading Seq2seq model instance to cloud for inferencing
            self.gcloud.sync_folder_to_gcloud(gcp_bucket_url=BUCKET_NAME, filepath=self.model_trainer_and_eval_config.seq_2_seq_model_to_gcp_path, filename=SEQ_2_SEQ_MODEL_NAME)
            logger.info(f"Seq2seq model object file uploaded to google container registry. File name - {SEQ_2_SEQ_MODEL_NAME}")

            model.apply(self.initialize_weights)
            logger.info("Weights applied to model")

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
                print(f'\tTrain Loss: {train_loss:.3f}')
                print(f'\t Val. Loss: {valid_loss:.3f}')

            model_trainer_artifacts = ModelTrainerAndEvalArtifacts(trained_model_path=self.model_trainer_and_eval_config.model_upload_path,
                                                                    source_vocab_file_path=self.model_trainer_and_eval_config.source_vocab_file_path,
                                                                    target_vocab_file_path=self.model_trainer_and_eval_config.target_vocab_file_path,
                                                                    seq_2_seq_model_path=self.model_trainer_and_eval_config.seq_2_seq_model_instance_path)

            logger.info("Exited the initiate_model_and_eval_trainer method of Model trainer and eval class")
            return model_trainer_artifacts

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e
