import os
#from datetime import datetime
from from_root import from_root


#TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_DIR = os.path.join(from_root(), "artifacts")#, TIMESTAMP)
LOGS_DIR = 'logs'
LOGS_FILE_NAME = 'code_generation.log' 
MODELS_DIR = 'models'

BUCKET_NAME = 'code-generation-using-transformers'
GCP_DATA_FILE_NAME = 'english_python_data.txt'
DATA_INGESTION_ARTIFACTS_DIR = 'DataIngestionArtifacts'