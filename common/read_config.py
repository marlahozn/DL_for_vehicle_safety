
import yaml
from pathlib import Path

class Config:

    @staticmethod
    def read_config(path: Path):
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config