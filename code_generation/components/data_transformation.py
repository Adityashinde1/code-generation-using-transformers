import os
import sys
import io
import random
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import keyword
import logging
import spacy
from tokenize import tokenize
from torchtext.legacy import data
from torchtext.legacy.data import Field, BucketIterator, Iterator
from code_generation.entity.config_entity import DataTransformationConfig
from code_generation.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts
from code_generation.exception import CodeGeneratorException
from code_generation.utils.main_utils import MainUtils
from code_generation.constants import *

logger = logging.getLogger(__name__)

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy.load('en_core_web_sm')


class DataTransformation:
    def __init__(self, data_ingestion_artifacts: DataIngestionArtifacts, data_transformation_config: DataTransformationConfig, utils: MainUtils) -> None:
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_transformation_config = data_transformation_config
        self.utils = utils


    def removing_hashtag(self, file_path: str) -> list:
        try:
            data = self.utils.read_txt_file(file_path=file_path)
            dps = []
            dp = None
            for line in data:
                if line[0] == "#":
                    if dp:
                        dp['solution'] = ''.join(dp['solution'])
                        dps.append(dp)
                    dp = {"question": None, "solution": []}
                    dp['question'] = line[1:]
                else:
                    dp["solution"].append(line)
            return dps

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

            train_expansion_factor = train_expansion_factor
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


    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            os.makedirs(self.data_transformation_config.data_transformation_artifacts_dir, exist_ok=True)
            logger.info(
                f"Created {os.path.basename(self.data_transformation_config.data_transformation_artifacts_dir)} directory."
            )    

            data_ = self.removing_hashtag(file_path=self.data_ingestion_artifacts.data_file_path) 
            logger.info("Removed # from the data.")

            data_df = pd.DataFrame(data_)
            train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE)
            logger.info("Splitted the data into train and test.")

            Input = data.Field(tokenize = 'spacy', init_token='', eos_token='', lower=True)
            Output = data.Field(tokenize = self.tokenize_python_code, init_token='', eos_token='', lower=False)

            fields = [('Input', Input),('Output', Output)]

            train_example, val_example = self.create_data_example(train_df=train_df, val_df=test_df, train_expansion_factor=TRAIN_EXPANSION_FACTOR, fields=fields)

            train_data = data.Dataset(train_example, fields)
            valid_data =  data.Dataset(val_example, fields)

            Input.build_vocab(train_data, min_freq = 0)
            Output.build_vocab(train_data, min_freq = 0)

            self.utils.dump_pickle_file(output_filepath=self.data_transformation_config.source_vocab_file_path, data=Input.vocab)
            self.utils.dump_pickle_file(output_filepath=self.data_transformation_config.target_vocab_file_path, data=Output.vocab)

            data_transformation_artifacts = DataTransformationArtifacts(source_vocab_file_path=self.data_transformation_config.source_vocab_file_path,
                                                                        target_vocab_file_path=self.data_transformation_config.target_vocab_file_path)

            return data_transformation_artifacts

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e