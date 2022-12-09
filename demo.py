from code_generation.components.data_ingestion import DataIngestion
from code_generation.entity.config_entity import DataIngestionConfig
from code_generation.configuration.gcloud import GCloud 

data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig(), gcloud=GCloud())

data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()