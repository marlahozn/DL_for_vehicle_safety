
import yaml

@staticmethod
def read_config(path: str):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config