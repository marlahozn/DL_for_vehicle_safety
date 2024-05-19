# Hyperas source: https://github.com/maxpumperla/hyperas

from __future__ import print_function
import numpy as np
import keras
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.layers.core import Dense, Activation
from keras.models import Sequential

from common.data_manager import DataManager

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """

    '''
    ######## 1) LOAD INPUT DATA ########
    '''
    mng = DataManager()
    inputs = mng.read_input_data()  # read info for the input parameters

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

    '''
    ######## 3) SPLIT TRAINING AND TESTING DATA ########
    '''
    n_train = 100
    x_train = inputs[:n_train]
    x_test = inputs[n_train:]
    y_train = output_curve[:n_train]
    y_test = output_curve[n_train:]

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """

    model = Sequential()
    model.add(Dense({{choice([8, 32, 64, 128, 256, 512, 1024])}}, input_shape=(x_train.shape[1],)))
    model.add(Activation({{choice(['linear', 'relu', 'tanh', 'sigmoid'])}}))
    model.add(Dense({{choice([8, 32, 64, 128, 256, 512, 1024])}}))
    model.add(Activation({{choice(['linear', 'relu', 'tanh', 'sigmoid'])}}))
    model.add(Dense({{choice([8, 32, 64, 128, 256, 512, 1024])}}))
    model.add(Activation({{choice(['linear', 'relu', 'tanh', 'sigmoid'])}}))
    model.add(Dense({{choice([8, 32, 64, 128, 256, 512, 1024])}}))
    model.add(Activation({{choice(['linear', 'relu', 'tanh', 'sigmoid'])}}))

    model.add(Dense(y_train.shape[1]))
    model.add(Activation({{choice(['linear', 'relu', 'tanh', 'sigmoid'])}}))

    adam = keras.optimizers.Adam(lr=10 ** -3)
    optim = adam

    model.compile(loss='mse', optimizer=optim)

    result = model.fit(x_train, y_train, batch_size=x_train.shape[0], epochs=350, verbose=2, validation_split=0.1)
    # Get the highest validation accuracy of the training epochs
    validation_loss = np.amin(result.history['val_loss'])
    print('Best validation loss of epoch:', validation_loss)
    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    # Run the function that finds the optimum hyperparameters from the defined spaces
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=2000, # total number of models to train before choosing the best one
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    np.save('y_moment_neck_best_model', best_run)
    np.save('y_moment_neck_best_model_error', best_model.evaluate(X_test, Y_test))