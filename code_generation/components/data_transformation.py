import os
import sys
import io
import random
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import keyword
import logging
from tokenize import tokenize
from torchtext.legacy import data
from code_generation.entity.config_entity import DataTransformationConfig
from code_generation.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts
from code_generation.exception import CodeGeneratorException
from code_generation.utils.main_utils import MainUtils
from code_generation.constants import *

logger = logging.getLogger(__name__)


class DataTransformation:
    def __init__(self, data_ingestion_artifacts = DataIngestionArtifacts, data_transformation_config = DataTransformationConfig, utils = MainUtils) -> None:
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts
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

            self.utils.dump_pickle_file(output_filepath=self.data_transformation_config.train_df_path, data=train_df)
            logger.info(f"Saved the train dataframe in data transformation artifacts directory. File name - {os.path.basename(self.data_transformation_config.train_df_path)}")

            self.utils.dump_pickle_file(output_filepath=self.data_transformation_config.test_df_path, data=test_df)
            logger.info(f"Saved the test dataframe in data transformation artifacts directory. File name - {os.path.basename(self.data_transformation_config.test_df_path)}")

            data_transformation_artifacts = DataTransformationArtifacts(train_df_path=self.data_transformation_config.train_df_path,
                                                                        test_df_path=self.data_transformation_config.test_df_path)

            return data_transformation_artifacts

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e
