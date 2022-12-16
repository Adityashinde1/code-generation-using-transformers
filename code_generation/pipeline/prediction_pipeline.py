import os
import sys
import logging
from code_generation.entity.config_entity import ModelPredictorConfig
from code_generation.exception import CodeGeneratorException
from code_generation.utils.main_utils import MainUtils
from code_generation.configuration.gcloud import GCloud
from code_generation.constants import *
from tokenize import untokenize
import spacy
import torch

logger = logging.getLogger(__name__)


class ModelPredictor:
    def __init__(self):
        self.model_predictor_config = ModelPredictorConfig()
        self.utils = MainUtils
        self.gcloud = GCloud()


    @staticmethod
    def generate_code(sentence, src_field, trg_field, model, device, max_len = 50000):
        try:
            model.eval()
                
            if isinstance(sentence, str):
                nlp = spacy.load('en_core_web_sm')
                tokens = [token.text.lower() for token in nlp(sentence)]
            else:
                tokens = [token.lower() for token in sentence]

            tokens = [src_field.init_token] + tokens + [src_field.eos_token]
                
            src_indexes = [src_field.vocab.stoi[token] for token in tokens]

            src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
            
            src_mask = model.make_src_mask(src_tensor)
            
            with torch.no_grad():
                enc_src = model.encoder(src_tensor, src_mask)

            trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

            for i in range(max_len):

                trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

                trg_mask = model.make_trg_mask(trg_tensor)
                
                with torch.no_grad():
                    output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                
                pred_token = output.argmax(2)[:,-1].item()
                
                trg_indexes.append(pred_token)

                if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                    break
            
            trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
            
            return trg_tokens[1:], attention

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    def create_python_code(self, src, SRC, TRG, model, device):
        try:
            src=src.split(" ")
            translation, attention = self.generate_code(src, SRC, TRG, model, device)
            code = untokenize(translation[:-1]).decode('utf-8')
            return code

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    def run_pipeline(self, src: str):
        try:

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.gcloud.sync_folder_from_gcloud(gcp_bucket_url=BUCKET_NAME, filename=SOURCE_VOCAB_FILE_NAME, destination=self.model_predictor_config.source_vocab_file_dest_path)
            self.gcloud.sync_folder_from_gcloud(gcp_bucket_url=BUCKET_NAME, filename=TARGET_VOCAB_FILE_NAME, destination=self.model_predictor_config.target_vocab_file_dest_path)

            SRC = self.utils.load_pickle_file(filepath=self.model_predictor_config.downloaded_source_vocab_file_path)
            TRG = self.utils.load_pickle_file(filepath=self.model_predictor_config.downloaded_target_vocab_file_path)

            self.gcloud.sync_folder_from_gcloud(gcp_bucket_url=BUCKET_NAME, filename=MODEL_FILE_NAME, destination=self.model_predictor_config.gcp_model_path)
            self.gcloud.sync_folder_from_gcloud(gcp_bucket_url=BUCKET_NAME, filename=SEQ_2_SEQ_MODEL_NAME, destination=self.model_predictor_config.seq_2_seq_model_instance_dest_path)

            model = torch.load(self.model_predictor_config.seq_2_seq_model_instance_path)

            model.load_state_dict(torch.load(self.model_predictor_config.best_model_path))

            code = self.create_python_code(src=src, SRC=SRC, TRG=TRG, model=model, device=device)

            return code

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e