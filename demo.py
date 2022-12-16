# from code_generation.components.data_ingestion import DataIngestion
# from code_generation.components.data_transformation import DataTransformation
# from code_generation.components.model_trainer_and_eval import ModelTrainerAndEval
# from code_generation.components.model_pusher import ModelPusher
# from code_generation.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerAndEvalConfig, ModelPusherConfig
# from code_generation.configuration.gcloud import GCloud 
# from code_generation.utils.main_utils import MainUtils




# data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig(), gcloud=GCloud())

# data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()

# data_transformation = DataTransformation(data_ingestion_artifacts=data_ingestion_artifacts, data_transformation_config=DataTransformationConfig(), utils=MainUtils)

# data_transformation_artifacts = data_transformation.initiate_data_transformation()

# model_trainer = ModelTrainerAndEval(data_transformation_artifacts=data_transformation_artifacts, model_trainer_and_eval_config=ModelTrainerAndEvalConfig(), utils=MainUtils, gcloud=GCloud())

# model_trainer_and_eval_artifacts = model_trainer.initiate_model_trainer()

# model_pusher = ModelPusher(model_trainer_and_eval_artifacts=model_trainer_and_eval_artifacts, model_pusher_config=ModelPusherConfig(), gcloud=GCloud())

# model_pusher.initiate_model_pusher()







from code_generation.pipeline.prediction_pipeline import ModelPredictor

pred = ModelPredictor()

code = pred.run_pipeline(src='write a function to add elements of list')

print(code)

# from code_generation.configuration.gcloud import GCloud
# from code_generation.constants import *
# from code_generation.entity.config_entity import ModelPredictorConfig
# gcloud = GCloud()
# model_predictor_config = ModelPredictorConfig()

# gcloud.sync_folder_from_gcloud(gcp_bucket_url=BUCKET_NAME, filename=MODEL_FILE_NAME, destination=model_predictor_config.gcp_model_path)