import sys
from code_generation.components.data_ingestion import DataIngestion
from code_generation.components.data_transformation import DataTransformation
from code_generation.components.model_trainer_and_eval import ModelTrainerAndEval
from code_generation.components.model_pusher import ModelPusher
from code_generation.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerAndEvalConfig, ModelPusherConfig
from code_generation.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts, ModelTrainerAndEvalArtifacts
from code_generation.entity.artifacts_entity import DataIngestionArtifacts
from code_generation.configuration.gcloud import GCloud
from code_generation.utils.main_utils import MainUtils
from code_generation.exception import CodeGeneratorException
from code_generation.constants import *
import logging

# initializing logger
logger = logging.getLogger(__name__)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_and_eval_config = ModelTrainerAndEvalConfig()
        self.model_pusher_config = ModelPusherConfig()
        self.gcloud = GCloud()


    # This method is used to start the data ingestion
    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logger.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logger.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config, gcloud=self.gcloud)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info("Got the train_set and test_set from mongodb")
            logger.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    # This method is used to start the data transformation
    def start_data_transformation(
        self, data_ingestion_artifact: DataIngestionArtifacts
    ) -> DataTransformationArtifacts:
        logger.info("Entered the start_data_preprocessing method of TrainPipeline class")
        try:
            data_transformation = DataTransformation(data_ingestion_artifacts=data_ingestion_artifact, data_transformation_config=self.data_transformation_config, utils=MainUtils)

            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logger.info("Performed the data validation operation")
            logger.info(
                "Exited the start_data_preprocessing method of TrainPipeline class"
            )
            return data_transformation_artifact

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    # This method is used to start the model trainer
    def start_model_trainer_and_eval(
        self, data_transformation_artifact: DataTransformationArtifacts) -> ModelTrainerAndEvalArtifacts:
        try:
            model_trainer = ModelTrainerAndEval(
                data_transformation_artifacts=data_transformation_artifact,
                model_trainer_and_eval_config=self.model_trainer_and_eval_config,
                utils=MainUtils,
                gcloud=self.gcloud
            )
            model_trainer_and_eval_artifact = model_trainer.initiate_model_and_eval_trainer()
            return model_trainer_and_eval_artifact

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e


    # This method is used to start the model pusher
    def start_model_pusher(
        self,
        model_trainer_and_eval_artifact: ModelTrainerAndEvalArtifacts
    ) -> None:
        logger.info("Entered the start_model_pusher method of TrainPipeline class")
        try:
            model_pusher = ModelPusher(
                model_trainer_and_eval_artifacts=model_trainer_and_eval_artifact,
                model_pusher_config=self.model_pusher_config,
                gcloud=self.gcloud
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logger.info("Initiated the model pusher")
            logger.info("Exited the start_model_pusher method of TrainPipeline class")
            return model_pusher_artifact

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e  


    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact) 
            model_trainer_and_eval_artifact = self.start_model_trainer_and_eval(data_transformation_artifact=data_transformation_artifact)
            model_pusher_artifact = self.start_model_pusher(model_trainer_and_eval_artifact=model_trainer_and_eval_artifact)

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e