import os
import pickle
from time import time

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.models import load_model

if __name__ == '__main__':

    #base_path = 'C:/0-MAR/Python/New_Data_for_Mar/'
    inputs = pd.read_csv(os.path.join(os.getcwd(), 'New_Data_for_Mar', 'Input_File.par'), header=0)
    inputs.columns = ['id', 'LHC_PAB', 'LHC_KAB', 'LHC_TTF', 'LHC_Belt']
    inputs = inputs.drop(['id'], axis=1)
    inputs = np.round(inputs.values, 6)  # 6th digits (since for simulations it was rounded)

    '''
    fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    parts = ax2.violinplot(df.values, showextrema=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightgray')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolor('k')
        vp.set_linewidth(1)
    # ax= plt.violinplot(df.values, showmedians=True)
    labels = ['x1', 'x2', 'x3', 'x4']
    ax2.set_xticks(np.arange(1, len(labels) + 1))
    ax2.set_xticklabels(labels)
    ax2.set_xlabel('Input parameters from mechanism-based model')
    plt.show()
    '''

    # Data set size
    n_sim, n_param = inputs.shape


    def select_data(selection):
        if (selection ==1):
            # x_acc_chest
            output_curve = pd.read_csv(os.path.join(os.getcwd(), 'New_Data_for_Mar', '13CHST0000H3ACXC_Curve.csv'), header=None)
            curve_name = 'x_acc_chest'
        elif(selection ==2):
            # x_force_neck
            output_curve = pd.read_csv(os.path.join(os.getcwd(), 'New_Data_for_Mar', '13NECKUP00H3FOXA_Curve.csv'), header=None)
            curve_name = 'x_force_neck'
        elif (selection == 3):
            # y_acc_chest
            output_curve = pd.read_csv(os.path.join(os.getcwd(), 'New_Data_for_Mar', '13CHST0000H3ACYA_Curve.csv'), header=None)
            curve_name = 'y_acc_chest'
        elif (selection == 4):
            # y_moment_neck
            output_curve = pd.read_csv(os.path.join(os.getcwd(), 'New_Data_for_Mar', '13NECKUP00H3MOYA_Curve.csv'), header=None)
            curve_name = 'y_moment_neck'
        elif (selection == 5):
            # y_moment_tibia
            output_curve = pd.read_csv(os.path.join(os.getcwd(), 'New_Data_for_Mar', '13TIBILEUPH3MOYB_Curve.csv'), header=None)
            curve_name = 'y_moment_tibia'
        elif (selection == 6):
            # y_vel_pelvic
            output_curve = pd.read_csv(os.path.join(os.getcwd(), 'New_Data_for_Mar', '13PELV0000H3VEYA_Curve.csv'), header=None)
            curve_name = 'y_vel_pelvic'
        elif (selection == 7):
            # z_force_neck
            output_curve = pd.read_csv(os.path.join(os.getcwd(), 'New_Data_for_Mar', '13NECKUP00H3FOZA_Curve.csv'), header=None)
            curve_name = 'z_force_neck'
        elif (selection == 8):
            # z_force_tibia
            output_curve = pd.read_csv(os.path.join(os.getcwd(), 'New_Data_for_Mar', '13TIBILELOH3FOZB_Curve.csv'), header=None)
            curve_name = 'z_force_tibia'
        elif (selection == 9):
            # z_vel_head
            output_curve = pd.read_csv(os.path.join(os.getcwd(), 'New_Data_for_Mar', '13HEAD0000H3VEZA_Curve.csv'), header=None)
            curve_name = 'z_vel_head'
        elif (selection == 10):
            # intru_chest
            output_curve = pd.read_excel(os.path.join(os.getcwd(), 'New_Data_for_Mar', 'intrusion_chest.xlsx'), header=None)#, engine='openpyxl')
            curve_name = 'intru_chest'
        else:
            # res_acc_head
            output_curve = pd.read_excel(os.path.join(os.getcwd(), 'New_Data_for_Mar', 'acc_res_head.xlsx'), header=None) #, engine='openpyxl')
            curve_name = 'res_acc_head'
        return output_curve, curve_name

    output_curve, output_name = select_data(4)

    min_value = np.min(output_curve.to_numpy())
    max_value = np.max(output_curve.to_numpy())

    #output_trans = (min_value - np.array(output_curve)) / (min_value - max_value)

    '''
    plt.figure()
    for i in range(0, 100):
        plt.plot(np.arange(0, 1150)/1000, output_plot[i, :])
    plt.xlabel('Time')
    plt.ylabel('Resultant head acceleration')
    plt.ylim((np.min(output_plot) - 0.05, np.max(output_plot) + 0.05))
    # plt.title('Validation curves (first samples)')
    #plt.show()
    '''

    n_timeSteps = output_curve.shape[1]

    # Scale Output Data
    #scaler = MinMaxScaler(feature_range=(-1, 1))  # StandardScaler()
    #output_trans = scaler.fit_transform(output_curve)
    output_trans = (output_curve.to_numpy() - output_curve.to_numpy().mean()) / output_curve.to_numpy().std()

    # SPLIT TRAINING AND TESTING DATA
    n_train = 100
    xIn_train = inputs[:n_train]
    xIn_val = inputs[n_train:]

    xOut_train = output_trans[:n_train]
    xOut_val = output_trans[n_train:]

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, DotProduct, WhiteKernel, RationalQuadratic, \
    ExpSineSquared, Matern

    # ONE MODEL FOR EACH TIME STEP
    def define_models(n_input, n_output):
        '''
        input_params = Input(shape=(n_input,))
        init = Dense(n_units, activation='tanh')(input_params)
        out_dense = Dense(n_output)(init)
        model = Model(input_params, out_dense)
        model.compile(optimizer='adam', loss='mse')
        '''

        model = LinearRegression()
        #model = KNeighborsRegressor()
        #kernel = DotProduct() + WhiteKernel()
        #kernel = RationalQuadratic(length_scale=1.0, alpha=1.5)
        #model = GaussianProcessRegressor(kernel=kernel,random_state=0)
        #model = GaussianProcessRegressor()  # default kernel is RBF
        #kernel = Matern(length_scale=1.0, nu=1.5)
        #model = GaussianProcessRegressor(kernel=kernel)
        #model = LinearSVR()
        #model = SVR(kernel='poly', gamma = 'auto')
        #model = DecisionTreeRegressor()
        #model = ElasticNet(alpha=1.0, l1_ratio=0.5)
        return model

    # configure problem
    n_features = 4
    n_steps_out = 1
    n_timeSteps_prev = 4

    '''
    # Store information of the model for each time step
    global_info = pd.DataFrame(index=range(n_timeSteps), columns={'selected_model_name', 'selected_model_ind', 'r2', 'predictions', 'predictions_training'})

    start_time = time()
    baseline = np.zeros((n_sim-n_train, n_timeSteps))
    baselineTRAIN = np.zeros((n_train, n_timeSteps))
    for i in range(n_timeSteps_prev+1): #for the first time steps we consider always the 4 input model
        # create model
        model_4 = define_models(n_param, n_steps_out)
        #dense_4model.compile(optimizer='adam', loss='mse')
        hist1 = model_4.fit(xIn_train, xOut_train[:, i])
        global_info.iloc[i]['selected_model_name'] = 'x1x2x3x4'
        global_info.iloc[i]['selected_model_ind'] = 0
        predictions0 = model_4.predict(xIn_val)
        predictions0_Train = model_4.predict(xIn_train)
        SSres = np.sum((predictions0 - xOut_val[:, i]) ** 2)
        SStot = np.sum((xOut_val[:, i] - np.mean(xOut_val[:, i])) ** 2)
        R2_all = 1 - (SSres / (SStot+(10e-20)))
        global_info.iloc[i]['r2'] = R2_all
        global_info.iloc[i]['predictions'] = predictions0
        global_info.iloc[i]['predictions_training'] = predictions0_Train
        baseline[:,i] = predictions0
        baselineTRAIN[:,i] = predictions0_Train
        # store the whole model info for prediction step
        #filename0 = os.path.join(os.getcwd(), 'metamodels','finalized_model_' + output_name)
        pickle.dump(model_4, open(os.path.join(os.getcwd(), 'metamodels_'+ output_name,'finalized_model_' + output_name + str(i) + '.sav'), 'wb'))
    '''

    def create_ínput_dict(i, xIn, xOut):
        inputs_dict = {'x1': xIn[:, 0].reshape(xIn.shape[0], 1),
                       'x2': xIn[:, 1].reshape(xIn.shape[0], 1),
                       'x3': xIn[:, 2].reshape(xIn.shape[0], 1),
                       'x4': xIn[:, 3].reshape(xIn.shape[0], 1),
                       'x1x2': np.vstack((xIn[:, 0], xIn[:, 1])).T,
                       'x1x3': np.vstack((xIn[:, 0], xIn[:, 2])).T,
                       'x1x4': np.vstack((xIn[:, 0], xIn[:, 3])).T,
                       'x2x3': np.vstack((xIn[:, 1], xIn[:, 2])).T,
                       'x2x4': np.vstack((xIn[:, 1], xIn[:, 3])).T,
                       'x3x4': np.vstack((xIn[:, 2], xIn[:, 3])).T,
                       'x1x2x3': np.vstack((xIn[:, 0], xIn[:, 1], xIn[:, 2])).T,
                       'x1x2x4': np.vstack((xIn[:, 0], xIn[:, 1], xIn[:, 3])).T,
                       'x1x3x4': np.vstack((xIn[:, 0], xIn[:, 2], xIn[:, 3])).T,
                       'x2x3x4': np.vstack((xIn[:, 1], xIn[:, 2], xIn[:, 3])).T,
                       'x1x2x3x4': xIn,
                       'x1+t1': np.vstack((xIn[:, 0], xOut[:, i - 1])).T,
                       'x1+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xOut[:, i-2:i].reshape(xOut.shape[0], 2))),
                       'x1+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xOut[:, i-3:i].reshape(xOut.shape[0], 3))),
                       'x1+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xOut[:, i-4:i].reshape(xOut.shape[0], 4))),
                       'x2+t1': np.vstack((xIn[:, 1], xOut[:, i - 1])).T,
                       'x2+t1t2': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x2+t1t2t3': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x2+t1t2t3t4': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x3+t1': np.vstack((xIn[:, 2], xOut[:, i - 1])).T,
                       'x3+t1t2': np.hstack((xIn[:, 2].reshape(xIn.shape[0], 1), xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x3+t1t2t3': np.hstack((xIn[:, 2].reshape(xIn.shape[0], 1), xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x3+t1t2t3t4': np.hstack((xIn[:, 2].reshape(xIn.shape[0], 1), xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x4+t1': np.vstack((xIn[:, 3], xOut[:, i - 1])).T,
                       'x4+t1t2': np.hstack((xIn[:, 3].reshape(xIn.shape[0], 1), xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x4+t1t2t3': np.hstack((xIn[:, 3].reshape(xIn.shape[0], 1), xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x4+t1t2t3t4': np.hstack((xIn[:, 3].reshape(xIn.shape[0], 1), xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x1x2+t1': np.vstack((xIn[:, 0], xIn[:, 1], xOut[:, i - 1])).T,
                       'x1x2+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), xOut[:, i-2:i].reshape(xOut.shape[0], 2))),
                       'x1x2+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1),xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x1x2+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x1x3+t1': np.vstack((xIn[:, 0], xIn[:, 2],xOut[:, i - 1])).T,
                       'x1x3+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x1x3+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1), xOut[:, i-3:i].reshape(xOut.shape[0], 3))),
                       'x1x3+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x1x4+t1': np.vstack((xIn[:, 0], xIn[:, 3], xOut[:, i - 1])).T,
                       'x1x4+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                              xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x1x4+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x1x4+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                  xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x2x3+t1': np.vstack((xIn[:, 1], xIn[:, 2], xOut[:, i - 1])).T,
                       'x2x3+t1t2': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                              xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x2x3+t1t2t3': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                                xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x2x3+t1t2t3t4': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                                  xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x2x4+t1': np.vstack((xIn[:, 1], xIn[:, 3], xOut[:, i - 1])).T,
                       'x2x4+t1t2': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                              xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x2x4+t1t2t3': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x2x4+t1t2t3t4': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                  xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x3x4+t1': np.vstack((xIn[:, 2], xIn[:, 3], xOut[:, i - 1])).T,
                       'x3x4+t1t2': np.hstack((xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                              xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x3x4+t1t2t3': np.hstack((xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x3x4+t1t2t3t4': np.hstack((xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                  xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x1x2x3+t1': np.vstack((xIn[:, 0], xIn[:, 1], xIn[:, 2], xOut[:, i - 1])).T,
                       'x1x2x3+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                                 xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x1x2x3+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                                xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x1x2x3+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                                  xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x1x2x4+t1': np.vstack((xIn[:, 0], xIn[:, 1], xIn[:, 3], xOut[:, i - 1])).T,
                       'x1x2x4+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x1x2x4+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                  xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x1x2x4+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                    xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x1x3x4+t1': np.vstack((xIn[:, 0], xIn[:, 2], xIn[:, 3], xOut[:, i - 1])).T,
                       'x1x3x4+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x1x3x4+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                  xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x1x3x4+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                    xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x2x3x4+t1': np.vstack((xIn[:, 1], xIn[:, 2], xIn[:, 3], xOut[:, i - 1])).T,
                       'x2x3x4+t1t2': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x2x3x4+t1t2t3': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                  xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x2x3x4+t1t2t3t4': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                    xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x1x2x3x4+t1': np.append(xIn[:,:], xOut[:, i-1].reshape(xOut.shape[0],1), axis=1),
                       'x1x2x3x4+t1t2': np.hstack((xIn,xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x1x2x3x4+t1t2t3': np.hstack((xIn,xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x1x2x3x4+t1t2t3t4': np.hstack((xIn, xOut[:, i - 4:i].reshape(xOut.shape[0], 4)))}
        return inputs_dict

    def create_val_dict(i, xIn, xOut):
        val_dict = {'x1': xIn[:, 0].reshape(xIn.shape[0], 1),
                       'x2': xIn[:, 1].reshape(xIn.shape[0], 1),
                       'x3': xIn[:, 2].reshape(xIn.shape[0], 1),
                       'x4': xIn[:, 3].reshape(xIn.shape[0], 1),
                       'x1x2': np.vstack((xIn[:, 0], xIn[:, 1])).T,
                       'x1x3': np.vstack((xIn[:, 0], xIn[:, 2])).T,
                       'x1x4': np.vstack((xIn[:, 0], xIn[:, 3])).T,
                       'x2x3': np.vstack((xIn[:, 1], xIn[:, 2])).T,
                       'x2x4': np.vstack((xIn[:, 1], xIn[:, 3])).T,
                       'x3x4': np.vstack((xIn[:, 2], xIn[:, 3])).T,
                       'x1x2x3': np.vstack((xIn[:, 0], xIn[:, 1], xIn[:, 2])).T,
                       'x1x2x4': np.vstack((xIn[:, 0], xIn[:, 1], xIn[:, 3])).T,
                       'x1x3x4': np.vstack((xIn[:, 0], xIn[:, 2], xIn[:, 3])).T,
                       'x2x3x4': np.vstack((xIn[:, 1], xIn[:, 2], xIn[:, 3])).T,
                       'x1x2x3x4': xIn,
                       'x1+t1': np.vstack((xIn[:, 0], xOut.iloc[i - 1]['predictions'])).T,
                       'x1+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), np.array([xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],2)))),
                       'x1+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1),np.array([xOut.iloc[i-3]['predictions'],
                                                                                xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],3)))),
                       'x1+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), np.array([xOut.iloc[i-4]['predictions'],
                                                                                xOut.iloc[i-3]['predictions'],
                                                                                xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],4)))),
                       'x2+t1': np.vstack((xIn[:, 1], xOut.iloc[i - 1]['predictions'])).T,
                       'x2+t1t2': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), np.array([xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],2)))),
                       'x2+t1t2t3': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), np.array([xOut.iloc[i-3]['predictions'],
                                                                                xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],3)))),
                       'x2+t1t2t3t4': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), np.array([xOut.iloc[i-4]['predictions'],
                                                                                xOut.iloc[i-3]['predictions'],
                                                                                xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],4)))),
                       'x3+t1': np.vstack((xIn[:, 2], xOut.iloc[i - 1]['predictions'])).T,
                       'x3+t1t2': np.hstack((xIn[:, 2].reshape(xIn.shape[0], 1), np.array([xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],2)))),
                       'x3+t1t2t3': np.hstack((xIn[:, 2].reshape(xIn.shape[0], 1), np.array([xOut.iloc[i-3]['predictions'],
                                                                                xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],3)))),
                       'x3+t1t2t3t4': np.hstack((xIn[:, 2].reshape(xIn.shape[0], 1), np.array([xOut.iloc[i-4]['predictions'],
                                                                                xOut.iloc[i-3]['predictions'],
                                                                                xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],4)))),
                       'x4+t1': np.vstack((xIn[:, 3], xOut.iloc[i - 1]['predictions'])).T,
                       'x4+t1t2': np.hstack((xIn[:, 3].reshape(xIn.shape[0], 1), np.array([xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],2)))),
                       'x4+t1t2t3': np.hstack((xIn[:, 3].reshape(xIn.shape[0], 1), np.array([xOut.iloc[i-3]['predictions'],
                                                                                xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],3)))),
                       'x4+t1t2t3t4': np.hstack((xIn[:, 3].reshape(xIn.shape[0], 1), np.array([xOut.iloc[i-4]['predictions'],
                                                                                xOut.iloc[i-3]['predictions'],
                                                                                xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],4)))),
                       'x1x2+t1': np.vstack((xIn[:, 0], xIn[:, 1], xOut.iloc[i - 1]['predictions'])).T,
                       'x1x2+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), np.array([xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],2)))),
                       'x1x2+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1),np.array([xOut.iloc[i-3]['predictions'],
                                                                                xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],3)))),
                       'x1x2+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), np.array([xOut.iloc[i-4]['predictions'],
                                                                                xOut.iloc[i-3]['predictions'],
                                                                                xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],4)))),
                       'x1x3+t1': np.vstack((xIn[:, 0], xIn[:, 2],xOut.iloc[i - 1]['predictions'])).T,
                       'x1x3+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),np.array([xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],2)))),
                       'x1x3+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1), np.array([xOut.iloc[i-3]['predictions'],
                                                                                xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],3)))),
                       'x1x3+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),np.array([xOut.iloc[i-4]['predictions'],
                                                                                xOut.iloc[i-3]['predictions'],
                                                                                xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],4)))),
                       'x1x4+t1': np.vstack((xIn[:, 0], xIn[:, 3], xOut.iloc[i - 1]['predictions'])).T,
                       'x1x4+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                               np.array([xOut.iloc[i - 2]['predictions'],
                                                         xOut.iloc[i - 1]['predictions']]).reshape(
                                                   (xOut_val.shape[0], 2)))),
                       'x1x4+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                 np.array([xOut.iloc[i - 3]['predictions'],
                                                           xOut.iloc[i - 2]['predictions'],
                                                           xOut.iloc[i - 1]['predictions']]).reshape(
                                                     (xOut_val.shape[0], 3)))),
                       'x1x4+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                   np.array([xOut.iloc[i - 4]['predictions'],
                                                             xOut.iloc[i - 3]['predictions'],
                                                             xOut.iloc[i - 2]['predictions'],
                                                             xOut.iloc[i - 1]['predictions']]).reshape(
                                                       (xOut_val.shape[0], 4)))),
                       'x2x3+t1': np.vstack((xIn[:, 1], xIn[:, 2], xOut.iloc[i - 1]['predictions'])).T,
                       'x2x3+t1t2': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                               np.array([xOut.iloc[i - 2]['predictions'],
                                                         xOut.iloc[i - 1]['predictions']]).reshape(
                                                   (xOut_val.shape[0], 2)))),
                       'x2x3+t1t2t3': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                                 np.array([xOut.iloc[i - 3]['predictions'],
                                                           xOut.iloc[i - 2]['predictions'],
                                                           xOut.iloc[i - 1]['predictions']]).reshape(
                                                     (xOut_val.shape[0], 3)))),
                       'x2x3+t1t2t3t4': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                                   np.array([xOut.iloc[i - 4]['predictions'],
                                                             xOut.iloc[i - 3]['predictions'],
                                                             xOut.iloc[i - 2]['predictions'],
                                                             xOut.iloc[i - 1]['predictions']]).reshape(
                                                       (xOut_val.shape[0], 4)))),
                       'x2x4+t1': np.vstack((xIn[:, 1], xIn[:, 3], xOut.iloc[i - 1]['predictions'])).T,
                       'x2x4+t1t2': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                               np.array([xOut.iloc[i - 2]['predictions'],
                                                         xOut.iloc[i - 1]['predictions']]).reshape(
                                                   (xOut_val.shape[0], 2)))),
                       'x2x4+t1t2t3': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                 np.array([xOut.iloc[i - 3]['predictions'],
                                                           xOut.iloc[i - 2]['predictions'],
                                                           xOut.iloc[i - 1]['predictions']]).reshape(
                                                     (xOut_val.shape[0], 3)))),
                       'x2x4+t1t2t3t4': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                   np.array([xOut.iloc[i - 4]['predictions'],
                                                             xOut.iloc[i - 3]['predictions'],
                                                             xOut.iloc[i - 2]['predictions'],
                                                             xOut.iloc[i - 1]['predictions']]).reshape(
                                                       (xOut_val.shape[0], 4)))),
                       'x3x4+t1': np.vstack((xIn[:, 2], xIn[:, 3], xOut.iloc[i - 1]['predictions'])).T,
                       'x3x4+t1t2': np.hstack((xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                               np.array([xOut.iloc[i - 2]['predictions'],
                                                         xOut.iloc[i - 1]['predictions']]).reshape(
                                                   (xOut_val.shape[0], 2)))),
                       'x3x4+t1t2t3': np.hstack((xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                 np.array([xOut.iloc[i - 3]['predictions'],
                                                           xOut.iloc[i - 2]['predictions'],
                                                           xOut.iloc[i - 1]['predictions']]).reshape(
                                                     (xOut_val.shape[0], 3)))),
                       'x3x4+t1t2t3t4': np.hstack((xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                   np.array([xOut.iloc[i - 4]['predictions'],
                                                             xOut.iloc[i - 3]['predictions'],
                                                             xOut.iloc[i - 2]['predictions'],
                                                             xOut.iloc[i - 1]['predictions']]).reshape(
                                                       (xOut_val.shape[0], 4)))),
                       'x1x2x3+t1': np.vstack((xIn[:, 0], xIn[:, 1], xIn[:, 2], xOut.iloc[i - 1]['predictions'])).T,
                       'x1x2x3+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                                 np.array([xOut.iloc[i - 2]['predictions'],
                                                           xOut.iloc[i - 1]['predictions']]).reshape(
                                                     (xOut_val.shape[0], 2)))),
                       'x1x2x3+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                                   np.array([xOut.iloc[i - 3]['predictions'],
                                                             xOut.iloc[i - 2]['predictions'],
                                                             xOut.iloc[i - 1]['predictions']]).reshape(
                                                       (xOut_val.shape[0], 3)))),
                       'x1x2x3+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                                     np.array([xOut.iloc[i - 4]['predictions'],
                                                               xOut.iloc[i - 3]['predictions'],
                                                               xOut.iloc[i - 2]['predictions'],
                                                               xOut.iloc[i - 1]['predictions']]).reshape(
                                                         (xOut_val.shape[0], 4)))),
                       'x1x2x4+t1': np.vstack((xIn[:, 0], xIn[:, 1], xIn[:, 3], xOut.iloc[i - 1]['predictions'])).T,
                       'x1x2x4+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                 np.array([xOut.iloc[i - 2]['predictions'],
                                                           xOut.iloc[i - 1]['predictions']]).reshape(
                                                     (xOut_val.shape[0], 2)))),
                       'x1x2x4+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                   np.array([xOut.iloc[i - 3]['predictions'],
                                                             xOut.iloc[i - 2]['predictions'],
                                                             xOut.iloc[i - 1]['predictions']]).reshape(
                                                       (xOut_val.shape[0], 3)))),
                       'x1x2x4+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                     np.array([xOut.iloc[i - 4]['predictions'],
                                                               xOut.iloc[i - 3]['predictions'],
                                                               xOut.iloc[i - 2]['predictions'],
                                                               xOut.iloc[i - 1]['predictions']]).reshape(
                                                         (xOut_val.shape[0], 4)))),
                       'x1x3x4+t1': np.vstack((xIn[:, 0], xIn[:, 2], xIn[:, 3], xOut.iloc[i - 1]['predictions'])).T,
                       'x1x3x4+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                 np.array([xOut.iloc[i - 2]['predictions'],
                                                           xOut.iloc[i - 1]['predictions']]).reshape(
                                                     (xOut_val.shape[0], 2)))),
                       'x1x3x4+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                   np.array([xOut.iloc[i - 3]['predictions'],
                                                             xOut.iloc[i - 2]['predictions'],
                                                             xOut.iloc[i - 1]['predictions']]).reshape(
                                                       (xOut_val.shape[0], 3)))),
                       'x1x3x4+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                     np.array([xOut.iloc[i - 4]['predictions'],
                                                               xOut.iloc[i - 3]['predictions'],
                                                               xOut.iloc[i - 2]['predictions'],
                                                               xOut.iloc[i - 1]['predictions']]).reshape(
                                                         (xOut_val.shape[0], 4)))),
                       'x2x3x4+t1': np.vstack((xIn[:, 1], xIn[:, 2], xIn[:, 3], xOut.iloc[i - 1]['predictions'])).T,
                       'x2x3x4+t1t2': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                 np.array([xOut.iloc[i - 2]['predictions'],
                                                           xOut.iloc[i - 1]['predictions']]).reshape(
                                                     (xOut_val.shape[0], 2)))),
                       'x2x3x4+t1t2t3': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                   np.array([xOut.iloc[i - 3]['predictions'],
                                                             xOut.iloc[i - 2]['predictions'],
                                                             xOut.iloc[i - 1]['predictions']]).reshape(
                                                       (xOut_val.shape[0], 3)))),
                       'x2x3x4+t1t2t3t4': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                     np.array([xOut.iloc[i - 4]['predictions'],
                                                               xOut.iloc[i - 3]['predictions'],
                                                               xOut.iloc[i - 2]['predictions'],
                                                               xOut.iloc[i - 1]['predictions']]).reshape(
                                                         (xOut_val.shape[0], 4)))),
                       'x1x2x3x4+t1': np.append(xIn[:,:], xOut.iloc[i-1]['predictions'].reshape(xOut_val.shape[0],1), axis=1),
                       'x1x2x3x4+t1t2': np.hstack((xIn,np.array([xOut.iloc[i-2]['predictions'],
                                                                 xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],2)))),
                       'x1x2x3x4+t1t2t3': np.hstack((xIn,np.array([xOut.iloc[i-3]['predictions'],
                                                                                xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],3)))),
                       'x1x2x3x4+t1t2t3t4': np.hstack((xIn, np.array([xOut.iloc[i-4]['predictions'],
                                                                                xOut.iloc[i-3]['predictions'],
                                                                                xOut.iloc[i-2]['predictions'],
                                                                                xOut.iloc[i-1]['predictions']]).reshape((xOut_val.shape[0],4))))}
        return val_dict

    n_models= 75
    '''
    for i in range(n_timeSteps_prev+1, n_timeSteps):
        inputs_dict = create_ínput_dict(i, xIn_train, xOut_train)
        val_dict = create_val_dict(i, xIn_val, global_info)
        # Store information of each of the 75 models
        model_selector = pd.DataFrame(index=range(n_models), columns={'predictions', 'predictions_training', 'r2'})
        for j in range(len(inputs_dict)):
            model_input = inputs_dict[list(inputs_dict)[j]]
            model = LinearRegression()
            model.fit(model_input, xOut_train[:, i])
            predictions = model.predict(val_dict[list(val_dict)[j]])
            predictions_train = model.predict(model_input)
            SSres = np.sum((xOut_val[:, i]-predictions) ** 2)
            SStot = np.sum((xOut_val[:, i] - np.mean(xOut_val[:, i])) ** 2)
            model_selector.iloc[j]['r2'] = 1 - (SSres/SStot)
            if j == 0:
                baseline[:, i] = predictions
                baselineTRAIN[:, i] = predictions_train
            model_selector.iloc[j]['predictions'] = predictions
            model_selector.iloc[j]['predictions_training'] = predictions_train
            SSres = 0
            SStot = 0

        min_loss = np.argmax([model_selector.iloc[:]['r2']])

        # SAVE everything related with the global model (i.e. the best model for each time step)
        model_input = inputs_dict[list(inputs_dict)[min_loss]]
        model = LinearRegression()
        model.fit(model_input, xOut_train[:, i])
        #filename1 = 'finalized_model' + output_name
        pickle.dump(model, open(os.path.join(os.getcwd(), 'metamodels_'+ output_name,'finalized_model_' + output_name + str(i) + '.sav'), 'wb'))

        global_info.iloc[i]['selected_model_name'] = list(val_dict)[min_loss]
        global_info.iloc[i]['selected_model_ind'] = min_loss
        global_info.iloc[i]['r2'] = model_selector.iloc[min_loss]['r2']
        global_info.iloc[i]['predictions'] = model_selector.iloc[min_loss]['predictions']
        global_info.iloc[i]['predictions_training'] = model_selector.iloc[min_loss]['predictions_training']

    train_time = time() - start_time
    np.save('training_time_' + output_name, train_time)

    #SAVE GLOBAL INFO
    global_info.to_csv('global_info_'+output_name+'.csv')
    '''

    global_info = pd.read_csv(open(os.path.join(os.getcwd(), 'Results', 'global_info_' + output_name + '.csv')), header=0)

    start_time = time()
    # prediction BASELINE: INDEPENDENT METAMODELS
    baseline = np.zeros((n_sim - n_train, n_timeSteps))
    for i in range(0, n_timeSteps):  # for the first time steps we consider always the 4 input model
        model_4 = define_models(n_param, n_steps_out)
        hist1 = model_4.fit(xIn_train, xOut_train[:, i])
        predictions0 = model_4.predict(xIn_val)
        baseline[:, i] = predictions0

    train_time = time() - start_time
    print('training_time_', train_time)

    predictionsAll = np.zeros((n_sim - n_train, n_timeSteps))
    #predictionsAllTRAIN = np.zeros((n_train, n_timeSteps))
    model_ind = np.zeros((n_timeSteps))
    name=[]
    for i in range(n_timeSteps):
        list_predictions = global_info.iloc[i]['predictions'][1:-1].split()
        predictions_transformat = []
        for j in range(xOut_val.shape[0]):
            predictions_transformat.append(float(list_predictions[j]))
        predictionsAll[:, i] = np.array(predictions_transformat).reshape((xOut_val.shape[0],))
        #predictionsAll[:, i] = global_info.iloc[i]['predictions'][1:-1].split().reshape((xOut_val.shape[0],))
        #predictionsAllTRAIN[:, i]= global_info.iloc[i]['predictions_training'].reshape((xOut_train.shape[0],))
        model_ind[i] = int(global_info.iloc[i]['selected_model_ind'])
        name.append(global_info.iloc[i]['selected_model_name'])

    n_time_steps = 1150

    model_dense = load_model('denses_OPT_'+output_name+'.h5') #el modelo es de 0,1
    ## PREDICTION
    predictions = model_dense.predict(xIn_val)
    # invert normalization of the dataset
    #predictionsOut = predictions*output_curve.to_numpy().std()+output_curve.to_numpy().mean()
    predictionsAll_75meta = predictionsAll*output_curve.to_numpy().std()+output_curve.to_numpy().mean()
    baseline_fin = baseline*output_curve.to_numpy().std()+output_curve.to_numpy().mean()
    val_out = xOut_val*output_curve.to_numpy().std()+output_curve.to_numpy().mean()

    #predictionsDense_real = (min_value - np.array(predictionsOut)) / (min_value - max_value)
    predictions_75meta_real = (min_value - np.array(predictionsAll_75meta)) / (min_value - max_value)
    baseline_real = (min_value - np.array(baseline_fin)) / (min_value - max_value)
    output_real = (min_value - np.array(val_out)) / (min_value - max_value)

    #savefig = savevariable + '40 comparision DENSE/'
    #REPRESENTATION
    for i in range(0, n_sim - n_train):
        plt.figure()
        plt.plot(np.arange(0, 1150) / 1000, output_real[i, :], 'darkblue')
        plt.plot(np.arange(0, 1150) / 1000, predictions[i, :], color='darkorange', ls='--')
        plt.plot(np.arange(0, 1150) / 1000, predictions_75meta_real[i, :], 'r')
        plt.plot(np.arange(0, 1150) / 1000, baseline_real[i, :], 'g-.')
        plt.xlabel('Time')
        plt.ylabel('Neck y-moment')
        plt.ylim((np.min(output_real) - 0.05, np.max(output_real) + 0.05))
        plt.legend(('Real', 'Dense', '75 metamodels', 'Baseline'))
        # plt.title('Output curves')
        plt.savefig('/Users/marlahoz/Desktop/TFM_BMW/new/Python/metamodels/Results/curves_' + output_name + '/' + str(
            i) + '.png')
        # plt.show()

    #for i in range(0, n_train):
        #plt.figure()
        #plt.plot(np.arange(0, 1150), xOut_train[i, :], 'darkblue')
        #plt.plot(np.arange(0, 1150), predictionsAllTRAIN[i, :], 'r')
        #plt.plot(np.arange(0, 1150), baselineTRAIN[i, :], 'g--')
        #plt.xlabel('Time (ms)')
        #plt.ylabel('Output curve')
        #plt.legend(('Real', 'Predicted', 'Baseline'))
        #plt.title('Output curves - Training')
        #plt.savefig(savefigTRAIN + str(i) + '.png')
        #plt.savefig(savefig + str(i) + '.png')
    

    #Barplot
    for i in range(n_timeSteps_prev+1, n_timeSteps):
        inputs_dict = create_ínput_dict(i, xIn_train, xOut_train)
    unique, counts = np.unique(model_ind, return_counts=True)
    #print(dict(zip(unique, counts)))
    global_count = np.zeros(75)
    for i in range(len(unique)):
        global_count[int(unique[i])]=counts[i]
    '''
    plt.figure()
    plt.bar(inputs_dict.keys(), global_count)
    plt.xticks(rotation=90)
    plt.title('Times each model is selected')
    plt.show()
    '''

    repres = np.round((global_count[:]/1150)*100,1)
    mar = []
    for i in range(len(repres)):
        mar.append((str(repres[i])+'%'))
    mar2 = np.array(mar)

    # Figure Size
    fig, ax = plt.subplots(figsize=(15, 16))
    # Horizontal Bar Plot
    df = pd.DataFrame({'0-0.6': repres, '0.8-0.1': repres1}, index=keys_list)
    keys_list = list(inputs_dict)
    ax = df.plot.barh()
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', which='major', pad=35)
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)
    ax.invert_yaxis()
    ax.set_yticklabels(keys_list, fontsize=10)
    keys_list = list(inputs_dict)
    for i in ax.patches:
        plt.text(i.get_width(), i.get_y(),
                 str(round((i.get_width()), 2)),
                 fontsize=10,
                 color='k')
    plt.show()

    df = pd.DataFrame({'0-0.6': repres, '0.8-0.1': repres1}, index = keys_list)
    ax = df.plot.barh()

    # library & dataset
    import seaborn as sns
    import matplotlib.pyplot as plt

    inputs = pd.read_csv(base_path + 'Input_File.par', header=0)
    inputs.columns = ['id', 'LHC_PAB', 'LHC_KAB', 'LHC_TTF', 'LHC_Belt']
    inputs = inputs.drop(['id'], axis=1)
    # Basic correlogram
    sns.pairplot(inputs, palette='Greens')
    plt.show()


    # ERROR DE PREDICCION sin tener en cuenta tiempo
    plt.figure()
    for i in range(0, n_sim - n_train):
        plt.scatter(predictions_75meta_real[i, :], output_real[i, :], s=0.5)
    plt.ylabel('Finite Element Simulations')
    plt.xlabel('75 metamodels Predicted values')
    # plt.title('Prediction error')
    plt.show()

    plt.figure()
    for i in range(0, n_sim - n_train):
        plt.scatter(predictionsDense_real[i, :], output_real[i, :], s=0.5)
    plt.ylabel('Finite Element Simulations')
    plt.xlabel('Optimized Feed-forward NN Predicted values')
    # plt.title('Prediction error')
    plt.show()

    ### REGRESSION EVALUATION METRICS
    # R2 SCORE
    # R2 SCORE - SKLEARN: https://stackoverflow.com/questions/33427849/significant-mismatch-between-r2-score-of-scikit-learn-and-the-r2-calculatio
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    print('R2-score-sklearn: ', r2_score(output_real.T, predictions_75meta_real.T))
    print('R2-score-sklearn-dense: ', r2_score(output_real.T, predictionsDense_real.T))
    print('R2-score-sklearn-Baseline: ', r2_score(output_real.T, baseline_real.T))

    # MEAN ABSOLUTE ERROR (MAE)

    print('MAE 75: ', mean_absolute_error(output_real, predictions_75meta_real))
    print('MAE dense: ', mean_absolute_error(output_real, predictionsDense_real))
    print('MAE baselin: ', mean_absolute_error(output_real, baseline_real))

    # MEAN SQUARED ERROR (MSE)

    print('MSE 75: ', mean_squared_error(output_real, predictions_75meta_real))
    print('MSE dense: ', mean_squared_error(output_real, predictionsDense_real))
    print('MSE baseline: ', mean_squared_error(output_real, baseline_real))

    print('CPU time: ', np.load((os.path.join(os.getcwd(), 'Results', 'training_time_' + output_name + '.npy'))))

    print('Fin')

    #fontweight = 'bold'