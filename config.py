from pathlib import Path
from dataclasses import dataclass

@dataclass()
class Config:
    dataset_path: Path = Path("data")
    model_storage_path: Path = Path("models")

    eda_condition: bool = False

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "Ames_Housing_Experiments"

    def __post_init__(self):
        self.train_data_path = self.dataset_path / "train.csv"
        self.test_data_path = self.dataset_path / "test.csv"

config = Config()