import os
import sys
import logging
from code_generation.entity.config_entity import DataIngestionConfig
from code_generation.entity.artifacts_entity import DataIngestionArtifacts
from code_generation.configuration.gcloud import GCloud
from code_generation.exception import CodeGeneratorException
from code_generation.constants import *


logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self,  data_ingestion_config: DataIngestionConfig, gcloud: GCloud) -> None:
        self.data_ingestion_config = data_ingestion_config
        self.gcloud = gcloud


    def get_data_from_gcp(self, bucket_name: str, file_name: str, path: str) -> None:
        logger.info("Entered the get_data_from_gcp method of data ingestion class")
        try:
            self.gcloud.sync_folder_from_gcloud(gcp_bucket_url=bucket_name, filename=file_name, destination=path)
            logger.info("Exited the get_data_from_gcp method of data ingestion class")

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e

    
    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logger.info("Entered the initiate_data_ingestion method of data ingestion class")
        try:
            # Creating Data Ingestion Artifacts directory inside artifacts folder
            os.makedirs(self.data_ingestion_config.data_ingestion_artifacts_dir, exist_ok=True)
            logger.info(
                f"Created {os.path.basename(self.data_ingestion_config.data_ingestion_artifacts_dir)} directory."
            )

            # Checking whether data file exists in the artifacts directory or not
            if os.path.exists(self.data_ingestion_config.gcp_data_file_path) == False:
                self.get_data_from_gcp(bucket_name=BUCKET_NAME, file_name=GCP_DATA_FILE_NAME, path=self.data_ingestion_config.gcp_data_file_path)
                logger.info(f"Downloaded the file from google cloud platform. File name - {os.path.basename(self.data_ingestion_config.gcp_data_file_path)}")

            data_ingestion_artifacts = DataIngestionArtifacts(data_file_path=self.data_ingestion_config.gcp_data_file_path)

            logger.info("Exited the initiate_data_ingestion method of data ingestion class")
            return data_ingestion_artifacts

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e