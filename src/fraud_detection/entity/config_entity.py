from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_version_dir: Path
    local_data_file: Path
    test_size: float
    random_state: int
    bucket_name: str

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    model_version_dir: Path
    data_version_dir: Path
    random_state: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    scale_pos_weight: int
    n_jobs: int
    objective: str

@dataclass(frozen=True)
class TrainingConfig:
    model_version_dir: Path
    data_version_dir: Path