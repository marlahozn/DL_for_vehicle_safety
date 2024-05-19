
from pathlib import Path
from common.data_manager import DataManager
from sklearn.metrics import mean_squared_error

from model.bayesianNN.BNN_model_FE import *
torch.manual_seed(0)

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
    ######## 4) BAYESIAN NEURAL NETWORK ########
    '''
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

    # TRAIN THE BNN MODEL
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper.lr)
    train_losses = np.zeros(hyper.max_epoch)

    # Training
    print('[INFO] Training network for ', hyper.max_epoch, 'epochs...')
    validation_MSE_total = []
    for epoch in range(hyper.max_epoch):
        train_loss = model.train(optimizer, X_train, Y_train, hyper)
        train_losses[epoch] = train_loss
        preds_tot = []
        for j in range(len(x_val)):
            preds = model.forward(X_val[j].reshape((-1, 1)))
            preds_tot.append(preds.data.cpu().numpy())
        validation_MSE = mean_squared_error(y_val.T, (np.array(preds_tot).squeeze(1)).T)
        print('Epoch: ', epoch, 'Train loss: ', train_losses[epoch], 'MSE validation: ', validation_MSE)
        validation_MSE_total.append(validation_MSE)

    print('[INFO] Training Ends! Saving model...')

    # Save the trained model
    output_folder = Path.home() / 'Desktop/paper_code/results/bayesianNN/'
    output_folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_name + '_Adam.pth')

