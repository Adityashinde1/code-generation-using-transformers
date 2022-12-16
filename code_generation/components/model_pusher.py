import sys
import logging
from code_generation.exception import CodeGeneratorException
from code_generation.entity.config_entity import ModelPusherConfig
from code_generation.entity.artifacts_entity import ModelTrainerAndEvalArtifacts, ModelPusherArtifacts
from code_generation.configuration.gcloud import GCloud
from code_generation.constants import *

logger = logging.getLogger(__name__)

class ModelPusher:
    def __init__(self, model_trainer_and_eval_artifacts: ModelTrainerAndEvalArtifacts, model_pusher_config: ModelPusherConfig, gcloud: GCloud) -> None:
        self.model_trainer_and_eval_artifacts = model_trainer_and_eval_artifacts
        self.model_pusher_config = model_pusher_config
        self.gcloud = gcloud


    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        logger.info("Entered the initiate_model_pusher method of Model pusher class")
        try:
            # Uploading the model to google container registry
            self.gcloud.sync_folder_to_gcloud(gcp_bucket_url=self.model_pusher_config.bucket_name,
                                                    filepath=self.model_trainer_and_eval_artifacts.trained_model_path,
                                                    filename=MODEL_FILE_NAME)
                                                    
            logger.info("Model pushed to google conatiner registry")
            model_pusher_artifacts = ModelPusherArtifacts(bucket_name=self.model_pusher_config.bucket_name,
                                                            gcp_model_path=self.model_trainer_and_eval_artifacts.trained_model_path)

            logger.info("Exited the initiate_model_pusher method of Model pusher class")
            return model_pusher_artifacts

        except Exception as e:
            raise CodeGeneratorException(e, sys) from e
