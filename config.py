from pathlib import Path
from dataclasses import dataclass

@dataclass()
class Config:
    dataset_path: Path = Path("data")
    model_storage_path: Path = Path("models")

    def __post_init__(self):
        self.train_data_path = self.dataset_path / "train.csv"
        self.test_data_path = self.dataset_path / "test.csv"

config = Config()