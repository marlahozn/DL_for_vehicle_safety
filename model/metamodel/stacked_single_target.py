# Author: Mar Lahoz Navarro
# December 9th, 2020
# Stacked single-target method
# This method builds an independent regression model for each time step in the output curve y

from common.data_manager import DataManager
from metamodel_manager import MetamodelManager

from pathlib import Path
import numpy as np
import pandas as pd

if __name__ == '__main__':

    '''
    ######## 1) LOAD INPUT DATA ########
    '''
    mng = DataManager()
    inputs = mng.read_input_data()  # read info for the input parameters
    n_sim, n_param = inputs.shape  # Input dataset size

    '''
    ######## 2) LOAD OUTPUT VARIABLE ######## 
    Possible output variables:
        - x_acc_chest, y_acc_chest
        - x_force_neck, z_force_neck
        - z_force_tibia
        - acc_res_head
        - z_vel_head
        - intrusion_chest
        - y_moment_neck
        - y_moment_tibia
        - y_vel_pelvic
    '''

    output_curve, output_name = mng.read_output_data(variable_name='x_acc_chest')
    n_timeSteps = output_curve.shape[1]

    '''
    ######## 3) SPLIT TRAINING AND TESTING DATA ########
    '''
    n_train = 100
    xIn_train = inputs[:n_train]
    xIn_val = inputs[n_train:]
    xOut_train = output_curve[:n_train]
    xOut_val = output_curve[n_train:]

    '''
    ######## 4) STACKED SINGLE-TARGET ########
    '''
    metamng = MetamodelManager()
    sst = np.zeros((n_sim - n_train, n_timeSteps))
    for i in range(0, n_timeSteps):  # for the first time steps we consider always the 4 input model
        model = metamng.define_model_for_each_time_step('LR')
        hist = model.fit(xIn_train, xOut_train[:, i])
        predictions = model.predict(xIn_val)
        sst[:, i] = predictions

    '''
    ######## 5) SAVE PREDICTIONS
    '''
    output_folder = Path.home() / 'Desktop/paper_code/results/stacked_single_target/'
    output_folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sst).to_csv(output_folder / (output_name + '.csv'))
