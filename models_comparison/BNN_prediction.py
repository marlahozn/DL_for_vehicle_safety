
from common.data_manager import DataManager
from pathlib import Path
import matplotlib.pyplot as plt
from model.bayesianNN.BNN_model_FE import *
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

    selected_curve = 'y_moment_neck'
    output_curve, output_name = mng.read_output_data(variable_name=selected_curve)
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
    ######## 4) LOAD MODEL ########
    '''
    model_folder = Path.home() / 'Desktop/paper_code/results/bayesianNN/'
    model_file = [file for file in model_folder.glob('*.pth') if output_name in file.stem]
    model_file = model_file[0]

    # Initialize hyperparameters
    hyper = BBB_Hyper(selected_dataset=selected_curve, validation_on=True)

    # CONVERT DATA TO TENSOR FORMAT
    Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype))
    if (hyper.validation):
        val_split = hyper.val_split
        x_train = x_train[:(int)(len(x_train) - len(x_train) * val_split), :]
        x_val = x_train[(int)(len(x_train) - len(x_train) * val_split):, :]
        y_train = y_train[:(int)(len(y_train) - len(y_train) * val_split), :]
        y_val = y_train[(int)(len(y_train) - len(y_train) * val_split):, :]

        n_input = x_train.shape[1]
        n_ouput = y_train.shape[1]

        X_train = Var(x_train)
        Y_train = Var(y_train)
        X_val = Var(x_val)
        Y_val = Var(y_val)
        X_test = Var(x_test)
        Y_test = Var(y_test)
    else:
        n_input = x_train.shape[1]
        n_ouput = y_train.shape[1]

        X_train = Var(x_train)
        Y_train = Var(y_train)
        X_test = Var(x_test)
        Y_test = Var(y_test)

    print('[INFO] Model hyperparameters:')
    print(hyper.__dict__)

    # Initialize network
    print('[INFO] Initializing network...')
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = BBB(n_input, n_ouput, hyper).cuda()
    else:
        model = BBB(n_input, n_ouput, hyper)

    # PREDICTION
    print('[INFO] Prediction time...')
    model.load_state_dict(torch.load(model_file))
    validation_points = 100
    total_pred_mean = []
    total_pred_std = []

    all_outputs = []
    for k in range(hyper.n_test_samples):
        outputs_SD = []
        for i in range(validation_points):
            x = X_test[k]  # 4 input params
            outputs_SD.append(model.forward(x.reshape((-1, 1))).data.cpu().numpy())
        predictions_SD = np.array(outputs_SD).squeeze(1)
        output_pred = predictions_SD
        all_outputs.append(output_pred)

    all_outputs = np.array(all_outputs)
    pred_mean = all_outputs.mean(1)  # Compute mean prediction
    pred_std = all_outputs.std(1)  # Compute standard deviation of prediction for each data point
    total_pred_mean = pred_mean
    total_pred_std = pred_std

    lower_limit_repre = np.min(np.array(total_pred_mean) - 3 * np.array(total_pred_std)) - 0.05
    upper_limit_repre = np.max(np.array(total_pred_mean) + 3 * np.array(total_pred_std)) + 0.05

    # EVALUATION METRICS
    print('MSE test data: ', mean_squared_error(y_test.T, np.array(total_pred_mean).T))
    print('MAE test data: ', mean_absolute_error(y_test.T, np.array(total_pred_mean).T))
    print('R2 test data: ', r2_score(y_test.T, np.array(total_pred_mean).T))

    # VISUALIZATION
    results_folder = Path.home() / 'Desktop/paper_code/results/bayesianNN/curves'
    results_folder.mkdir(parents=True, exist_ok=True)
    for k in range(hyper.n_test_samples):
        plt.fill_between(np.arange(0, n_timeSteps)/1000, total_pred_mean[k] - 3 * total_pred_std[k], total_pred_mean[k] + 3 * total_pred_std[k],
                         color='cornflowerblue', alpha=.5, label='+/- 3 std')
        plt.plot(np.arange(0, n_timeSteps)/1000, total_pred_mean[k], c='red', label='BNN Prediction')
        plt.plot(np.arange(0, n_timeSteps)/1000, y_test[k, :], c='black', label='Truth')
        plt.ylim((lower_limit_repre, upper_limit_repre))
        plt.xlabel('Time')
        plt.ylabel(selected_curve)
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_folder + str(k) + '.png')
        plt.clf()