import os
from from_root import from_root

ARTIFACTS_DIR = os.path.join(from_root(), "artifacts")#, TIMESTAMP)
LOGS_DIR = 'logs'
LOGS_FILE_NAME = 'code_generation.log' 


BUCKET_NAME = 'code-generation-using-transformers'
GCP_DATA_FILE_NAME = 'english_python_data.txt'
DATA_INGESTION_ARTIFACTS_DIR = 'DataIngestionArtifacts'

DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
TEST_SIZE = 0.15
TRAIN_EXPANSION_FACTOR = 100
TRAIN_DF_FILE_NAME = 'train_df.pkl'
TEST_DF_FILE_NAME = 'test_df.pkl'


MODEL_TRAINER_AND_EVAL_ARTIFACTS_DIR = 'ModelTrainerAndEvalArtifacts'
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 16
DEC_HEADS = 16
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
ENC_MAX_LENGTH = 1000
DEC_MAX_LENGTH = 10000
SOURCE_VOCAB_FILE_NAME = 'inpute_src.pkl'
TARGET_VOCAB_FILE_NAME = 'output_trg.pkl'
SEQ_2_SEQ_MODEL_NAME = 'seq_to_seq_model_instance.pt'

LEARNING_RATE = 0.0005
EPOCHS = 100
CLIP = 1
BEST_VALID_LOSS = float('inf')
BATCH_SIZE = 16

MODEL_FILE_NAME = 'model.pt'

APP_HOST = "0.0.0.0"
APP_PORT = 8080


