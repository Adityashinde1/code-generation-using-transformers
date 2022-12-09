from dataclasses import dataclass

# Data Ingestion Artifacts
@dataclass
class DataIngestionArtifacts:
    data_file_path: str


# Data Transformation Artifacts
@dataclass
class DataTransformationArtifacts:
    source_vocab_file_path: str
    target_vocab_file_path: str