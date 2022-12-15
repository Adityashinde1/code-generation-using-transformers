from dataclasses import dataclass

# Data Ingestion Artifacts
@dataclass
class DataIngestionArtifacts:
    data_file_path: str


# Data Transformation Artifacts
@dataclass
class DataTransformationArtifacts:
    train_df_path: str
    test_df_path: str


# Model Trainer Artifacts
@dataclass
class ModelTrainerAndEvalArtifacts:
    trained_model_path: str


# Model Pusher Artifacts
@dataclass
class ModelPusherArtifacts:
    bucket_name: str
    gcp_model_path: str
