from dataclasses import dataclass
from from_root import from_root
import os
from code_generation.constants import *


@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.data_ingestion_artifacts_dir: str = os.path.join(from_root(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.gcp_data_file_path: str = os.path.join(self.data_ingestion_artifacts_dir, GCP_DATA_FILE_NAME)

@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.data_transformation_artifacts_dir: str = os.path.join(from_root(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.train_df_path: str = os.path.join(self.data_transformation_artifacts_dir, TRAIN_DF_FILE_NAME)
        self.test_df_path: str = os.path.join(self.data_transformation_artifacts_dir, TEST_DF_FILE_NAME)
        self.source_vocab_file_path: str = os.path.join(self.data_transformation_artifacts_dir, SOURCE_VOCAB_FILE_NAME)
        self.target_vocab_file_path: str = os.path.join(self.data_transformation_artifacts_dir, TARGET_VOCAB_FILE_NAME)

@dataclass
class ModelTrainerAndEvalConfig:
    def __init__(self):
        self.model_trainer_and_eval_artifacts_dir: str = os.path.join(from_root(), ARTIFACTS_DIR, MODEL_TRAINER_AND_EVAL_ARTIFACTS_DIR)
        self.model_path: str = os.path.join(self.model_trainer_and_eval_artifacts_dir, MODEL_FILE_NAME)
        self.model_upload_path: str = os.path.join(self.model_trainer_and_eval_artifacts_dir)


@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.bucket_name: str = BUCKET_NAME
        self.model_name: str = MODEL_FILE_NAME
