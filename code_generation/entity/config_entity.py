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


@dataclass
class ModelTrainerAndEvalConfig:
    def __init__(self):
        self.model_trainer_and_eval_artifacts_dir: str = os.path.join(from_root(), ARTIFACTS_DIR, MODEL_TRAINER_AND_EVAL_ARTIFACTS_DIR)
        self.source_vocab_file_path: str = os.path.join(self.model_trainer_and_eval_artifacts_dir, SOURCE_VOCAB_FILE_NAME)
        self.source_file_to_gcp_path: str = os.path.join(self.model_trainer_and_eval_artifacts_dir)
        self.target_vocab_file_path: str = os.path.join(self.model_trainer_and_eval_artifacts_dir, TARGET_VOCAB_FILE_NAME)
        self.target_file_to_gcp_path: str = os.path.join(self.model_trainer_and_eval_artifacts_dir)
        self.model_path: str = os.path.join(self.model_trainer_and_eval_artifacts_dir, MODEL_FILE_NAME)
        self.model_upload_path: str = os.path.join(self.model_trainer_and_eval_artifacts_dir)
        self.seq_2_seq_model_instance_path: str = os.path.join(self.model_trainer_and_eval_artifacts_dir, SEQ_2_SEQ_MODEL_NAME)
        self.seq_2_seq_model_to_gcp_path: str = os.path.join(self.model_trainer_and_eval_artifacts_dir)


@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.bucket_name: str = BUCKET_NAME
        self.model_name: str = MODEL_FILE_NAME


@dataclass
class ModelPredictorConfig:
    def __init__(self):
        self.gcp_model_path: str = os.path.join(from_root())
        self.best_model_path: str = os.path.join(from_root(), MODEL_FILE_NAME)
        self.source_vocab_file_dest_path: str = os.path.join(from_root())
        self.downloaded_source_vocab_file_path: str = os.path.join(from_root(), SOURCE_VOCAB_FILE_NAME)
        self.target_vocab_file_dest_path: str = os.path.join(from_root())
        self.downloaded_target_vocab_file_path: str = os.path.join(from_root(), TARGET_VOCAB_FILE_NAME)
        self.seq_2_seq_model_instance_dest_path: str = os.path.join(from_root())
        self.seq_2_seq_model_instance_path: str = os.path.join(from_root(), SEQ_2_SEQ_MODEL_NAME)