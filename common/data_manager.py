
from pathlib import Path
import pandas as pd

class DataManager:

    INPUT_DATA_PATH = Path.home() / 'Desktop/paper_code' / 'data/input/'  # hardcoded data path for input file
    OUTPUT_DATA_PATH = Path.home() / 'Desktop/paper_code' / 'data/output/'  # hardcoded data path for output file

    def __init__(self):
        self.output_filenames = [file for file in self.OUTPUT_DATA_PATH.glob('*')]
        self.input_filename = [file for file in self.INPUT_DATA_PATH.glob('*.csv')]

    def read_output_data(self, variable_name: str):
        for file in self.output_filenames:
            if file.stem == variable_name:
                output_curve = pd.read_csv(file, header=0, index_col=[0])
                output_curve = output_curve.to_numpy()
                output_name = variable_name
        return output_curve, output_name

    def read_input_data(self):
        for file in self.input_filename:
            inputs = pd.read_csv(file, header=0, index_col=[0])
            inputs = inputs.to_numpy()
        return inputs






