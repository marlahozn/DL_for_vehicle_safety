# Author: Mar Lahoz Navarro
# December 9th, 2020
# Modified Ensemble of Regression Chains
# This method builds an independent regression model for each time step in the output curve y
# considering as input the combination of the input parameters and the n previous time steps
# With this method, the goal is to explore the importance of the previous time steps in predicting
# the future one

import os
import pickle

import common as c
from common.data_manager import DataManager
from metamodel_manager import MetamodelManager
from common.read_config import Config as c

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

    output_curve, output_name = mng.read_output_data(variable_name='y_moment_neck')
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
    ######## 4) READ MODEL CONFIGURATION ########
    '''
    config = c.read_config(Path.home() / 'Desktop/paper_code/config/m_ensemble_regressor_chains.yaml')
    config = config['mERC']
    n_steps_out = config['n_steps_out']
    n_timeSteps_prev = config['n_timeSteps_prev']
    n_models = config['n_model_combinations']

    '''
    ######## 5) MODIFIED ENSEMBLE OF REGRESSOR CHAINS ########
    '''
    # Store information of the model for each time step
    merc_all_info = pd.DataFrame(index=range(n_timeSteps),
                                 columns={'selected_model_name', 'selected_model_ind', 'r2', 'predictions_val',
                                          'predictions_train'})
    metamng = MetamodelManager()
    pred_val = np.zeros((n_sim - n_train, n_timeSteps))  # validation curves
    pred_train = np.zeros((n_train, n_timeSteps))

    # AS FOR THE FIRST DEFINED n_timeSteps_prev WE CAN ONLY CONSIDER THE 4 INPUTS PARAMETERS,
    # WE ARE GOING TO SPLIT THE TRAINING IN 2 DIFFERENT BLOCKS
    # FIRST BLOCK: TRAIN A MODEL FOR THE DEFINED n_timeSteps_prev
    for i in range(n_timeSteps_prev + 1):
        model = metamng.define_model_for_each_time_step('LR')  # create model
        model.fit(xIn_train, xOut_train[:, i])
        merc_all_info.iloc[i]['selected_model_name'] = 'x1x2x3x4'  # save info of the selected model
        merc_all_info.iloc[i]['selected_model_ind'] = 0
        predictions_val = model.predict(xIn_val)
        predictions_train = model.predict(xIn_train)
        SSres = np.sum((predictions_val - xOut_val[:, i]) ** 2)
        SStot = np.sum((xOut_val[:, i] - np.mean(xOut_val[:, i])) ** 2)
        R2_all = 1 - (SSres / (SStot + (10e-20)))  # add a constant to avoid the division by 0
        merc_all_info.iloc[i]['r2'] = R2_all
        merc_all_info.iloc[i]['predictions_val'] = predictions_val
        merc_all_info.iloc[i]['predictions_train'] = predictions_train
        pred_val[:, i] = predictions_val
        pred_train[:, i] = predictions_train

    # SECOND BLOCK: TRAIN A MODEL FOR EACH TIME STEP FROM n_timeSteps_prev TO n_timeSteps, WHERE WE CAN CONSIDER
    # ALL THE 75 INPUT COMBINATIONS
    for i in range(n_timeSteps_prev + 1, n_timeSteps):
        inputs_dict = metamng.create_ínput_combination_dict(i, xIn_train, xOut_train)
        val_dict = metamng.create_ínput_combination_dict(i, xIn_val, pred_val)
        # We are going to train 75 models for each time step (coming from the 75 inputs parameters combination)
        model_selector = pd.DataFrame(index=range(n_models), columns={'predictions_val', 'predictions_train', 'r2'})
        for j in range(len(inputs_dict)):  # run the model for each of the 75 input combinations
            model_input = inputs_dict[list(inputs_dict)[j]]
            model = metamng.define_model_for_each_time_step('LR')
            model.fit(model_input, xOut_train[:, i])
            predictions_val = model.predict(val_dict[list(val_dict)[j]])
            predictions_train = model.predict(model_input)
            SSres = np.sum((xOut_val[:, i] - predictions_val) ** 2)
            SStot = np.sum((xOut_val[:, i] - np.mean(xOut_val[:, i])) ** 2)
            model_selector.iloc[j]['r2'] = 1 - (SSres / SStot)
            if j == 0:
                pred_val[:, i] = predictions_val
                pred_train[:, i] = predictions_train
            model_selector.iloc[j]['predictions_val'] = predictions_val
            model_selector.iloc[j]['predictions_train'] = predictions_train

        # From the 75 models, we are going to select the best one
        min_loss = np.argmax([model_selector.iloc[:]['r2']])

        # Save the best model information
        merc_all_info.iloc[i]['selected_model_name'] = list(val_dict)[min_loss]
        merc_all_info.iloc[i]['selected_model_ind'] = min_loss
        merc_all_info.iloc[i]['r2'] = model_selector.iloc[min_loss]['r2']
        merc_all_info.iloc[i]['predictions_val'] = model_selector.iloc[min_loss]['predictions_val']
        merc_all_info.iloc[i]['predictions_train'] = model_selector.iloc[min_loss]['predictions_train']

    '''
    ######## 6) SAVE PREDICTIONS AND INFORMATION ########
    '''
    output_folder = Path.home() / 'Desktop/paper_code/results/m_ensemble_regressor_chains/'
    output_folder.mkdir(parents=True, exist_ok=True)
    merc_all_info.to_csv(output_folder / (output_name + '.csv'))
