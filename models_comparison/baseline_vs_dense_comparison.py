
from common.data_manager import DataManager
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

if __name__ == '__main__':

    '''
    ######## 1) LOAD INPUT DATA ########
    '''
    mng = DataManager()
    inputs = mng.read_input_data()  # read info for the input parameters
    n_sim, n_param = inputs.shape  # Input dataset size

    '''
    ######## 2) LOAD OUTPUT DATA ########
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
    mng = DataManager()
    output_curve, output_name = mng.read_output_data(variable_name='y_moment_neck')
    n_timeSteps = output_curve.shape[1]

    '''
    ######## 3) SPLIT TRAINING AND TESTING DATA ########
    '''
    n_train = 100
    x_train = inputs[:n_train]
    x_test = inputs[n_train:]
    y_train = output_curve[:n_train]
    y_test = output_curve[n_train:]

    '''
    ######## 4) LOAD ALL MODEL ########
    '''
    # Stacked single target
    output_folder = Path.home() / 'Desktop/paper_code/results/stacked_single_target/'
    model_file = [file for file in output_folder.glob('*.csv') if output_name in file.stem]
    sst_model_path = model_file[0]
    if sst_model_path.exists():
        sst_predictions = pd.read_csv(sst_model_path)
        sst_predictions = sst_predictions.drop(columns='Unnamed: 0').to_numpy()
    else:
        print('There is no sst model for ' + {output_name})
        exit()

    # Modified Ensemble of Regressor Chains
    output_folder = Path.home() / 'Desktop/paper_code/results/m_ensemble_regressor_chains/'
    model_file = [file for file in output_folder.glob('*.csv') if output_name in file.stem]
    merc_model_path = model_file[0]
    if merc_model_path.exists():
        merc_info = pd.read_csv(merc_model_path)
        merc_predictions = np.zeros((y_test.shape[0], n_timeSteps))
        for i in range(n_timeSteps):
            list_predictions = merc_info.iloc[i]['predictions'][1:-1].split()
            merc_preds = []
            for j in range(y_test.shape[0]):
                merc_preds.append(float(list_predictions[j]))
            merc_predictions[:, i] = np.array(merc_preds).reshape((y_test.shape[0],))
    else:
        print('There is no sst model for ' + {output_name})
        exit()

    # Dense NN
    output_folder = Path.home() / 'Desktop/paper_code/results/dense/'
    model_file = [file for file in output_folder.glob('*.h5') if output_name in file.stem]
    dense_model_path = model_file[0]
    if dense_model_path.exists():
        dense_model = load_model(dense_model_path)
        dense_predictions = dense_model.predict(x_test)
    else:
        print('There is no sst model for ' + {output_name})
        exit()

    '''
    ######## 5) PLOT RESULTS VS REAL CURVES ########
    '''
    results_folder = Path.home() / 'Desktop/paper_code/results/baseline_vs_dense_comparison/curves' / output_name
    results_folder.mkdir(parents=True, exist_ok=True)
    for i in range(0, y_test.shape[0]):
        plt.figure()
        plt.plot(np.arange(0, n_timeSteps) / 1000, y_test[i, :], 'darkblue')
        plt.plot(np.arange(0, n_timeSteps) / 1000, dense_predictions[i, :], color='darkorange', ls='--')
        plt.plot(np.arange(0, n_timeSteps) / 1000, merc_predictions[i, :], 'r')
        plt.plot(np.arange(0, n_timeSteps) / 1000, sst_predictions[i, :], 'g-.')
        plt.xlabel('Time')
        plt.ylabel('Neck y-moment')
        plt.ylim((np.min(y_test) - 0.05, np.max(y_test) + 0.05))
        plt.legend(('Real', 'Dense', '75 metamodels - MERC', 'Baseline - SST'))
        plt.savefig(results_folder / (str(i) + '.png'))

    '''
    ######## 6) EVALUATION METRICS ########
    '''
    # R2 SCORE:
    print('R2-score 75 MERC:  ', r2_score(y_test.T, merc_predictions.T))
    print('R2-score Dense: ', r2_score(y_test.T, dense_predictions.T))
    print('R2-score SST: ', r2_score(y_test.T, sst_predictions.T))

    # MEAN ABSOLUTE ERROR (MAE)
    print('MAE 75 MERC: ', mean_absolute_error(y_test, merc_predictions))
    print('MAE Dense: ', mean_absolute_error(y_test, dense_predictions))
    print('MAE SST: ', mean_absolute_error(y_test, sst_predictions))

    # MEAN SQUARED ERROR (MSE)
    print('MSE 75 MERC: ', mean_squared_error(y_test, merc_predictions))
    print('MSE Dense: ', mean_squared_error(y_test, dense_predictions))
    print('MSE SST: ', mean_squared_error(y_test, sst_predictions))

