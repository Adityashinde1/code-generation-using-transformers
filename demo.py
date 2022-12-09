from code_generation.components.data_ingestion import DataIngestion
from code_generation.components.data_transformation import DataTransformation
from code_generation.entity.config_entity import DataIngestionConfig, DataTransformationConfig
from code_generation.entity.artifacts_entity import DataIngestionArtifacts
from code_generation.configuration.gcloud import GCloud 
from code_generation.utils.main_utils import MainUtils



data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig(), gcloud=GCloud())

data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()

data_transformation = DataTransformation(data_ingestion_artifacts=data_ingestion_artifacts, data_transformation_config=DataTransformationConfig(), utils=MainUtils)

data_transformation_artifacts = data_transformation.initiate_data_transformation()

