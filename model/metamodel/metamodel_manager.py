
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.linear_model import LinearRegression

class MetamodelManager():

    def __init__(self):
        pass

    def define_model_for_each_time_step(self, selected_model: str):
        '''
        Select the model for each time step
        '''
        model = None
        if selected_model == 'LR':
            model = LinearRegression()
        elif selected_model == 'KNN':
            model = KNeighborsRegressor()
        elif selected_model == 'GP':
            model = GaussianProcessRegressor()  # default kernel is RBF
        elif selected_model == 'GP_kernel':
            kernel = RationalQuadratic(length_scale=1.0, alpha=1.5)
            model = GaussianProcessRegressor(kernel=kernel)
        else:
            print('Selected model ' + {selected_model} + 'is not defined')

        return model

    def create_ínput_combination_dict(self, i, xIn, xOut):
        '''
        The possible 75 combinations are hardcoded and a dict with all the combinations is returned
        :param i:
        :param xIn:
        :param xOut:
        :return:
        '''

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
                       'x1+t1t2': np.hstack(
                           (xIn[:, 0].reshape(xIn.shape[0], 1), xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x1+t1t2t3': np.hstack(
                           (xIn[:, 0].reshape(xIn.shape[0], 1), xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x1+t1t2t3t4': np.hstack(
                           (xIn[:, 0].reshape(xIn.shape[0], 1), xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x2+t1': np.vstack((xIn[:, 1], xOut[:, i - 1])).T,
                       'x2+t1t2': np.hstack(
                           (xIn[:, 1].reshape(xIn.shape[0], 1), xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x2+t1t2t3': np.hstack(
                           (xIn[:, 1].reshape(xIn.shape[0], 1), xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x2+t1t2t3t4': np.hstack(
                           (xIn[:, 1].reshape(xIn.shape[0], 1), xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x3+t1': np.vstack((xIn[:, 2], xOut[:, i - 1])).T,
                       'x3+t1t2': np.hstack(
                           (xIn[:, 2].reshape(xIn.shape[0], 1), xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x3+t1t2t3': np.hstack(
                           (xIn[:, 2].reshape(xIn.shape[0], 1), xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x3+t1t2t3t4': np.hstack(
                           (xIn[:, 2].reshape(xIn.shape[0], 1), xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x4+t1': np.vstack((xIn[:, 3], xOut[:, i - 1])).T,
                       'x4+t1t2': np.hstack(
                           (xIn[:, 3].reshape(xIn.shape[0], 1), xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x4+t1t2t3': np.hstack(
                           (xIn[:, 3].reshape(xIn.shape[0], 1), xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x4+t1t2t3t4': np.hstack(
                           (xIn[:, 3].reshape(xIn.shape[0], 1), xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x1x2+t1': np.vstack((xIn[:, 0], xIn[:, 1], xOut[:, i - 1])).T,
                       'x1x2+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1),
                                               xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x1x2+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1),
                                                 xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x1x2+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1),
                                                   xIn[:, 1].reshape(xIn.shape[0], 1),
                                                   xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x1x3+t1': np.vstack((xIn[:, 0], xIn[:, 2], xOut[:, i - 1])).T,
                       'x1x3+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                               xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x1x3+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                                 xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x1x3+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1),
                                                   xIn[:, 2].reshape(xIn.shape[0], 1),
                                                   xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x1x4+t1': np.vstack((xIn[:, 0], xIn[:, 3], xOut[:, i - 1])).T,
                       'x1x4+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                               xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x1x4+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                 xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x1x4+t1t2t3t4': np.hstack(
                           (xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                            xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x2x3+t1': np.vstack((xIn[:, 1], xIn[:, 2], xOut[:, i - 1])).T,
                       'x2x3+t1t2': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                               xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x2x3+t1t2t3': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                                 xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x2x3+t1t2t3t4': np.hstack(
                           (xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                            xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x2x4+t1': np.vstack((xIn[:, 1], xIn[:, 3], xOut[:, i - 1])).T,
                       'x2x4+t1t2': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                               xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x2x4+t1t2t3': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                 xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x2x4+t1t2t3t4': np.hstack(
                           (xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                            xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x3x4+t1': np.vstack((xIn[:, 2], xIn[:, 3], xOut[:, i - 1])).T,
                       'x3x4+t1t2': np.hstack((xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                               xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x3x4+t1t2t3': np.hstack((xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                                                 xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x3x4+t1t2t3t4': np.hstack(
                           (xIn[:, 2].reshape(xIn.shape[0], 1), xIn[:, 3].reshape(xIn.shape[0], 1),
                            xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x1x2x3+t1': np.vstack((xIn[:, 0], xIn[:, 1], xIn[:, 2], xOut[:, i - 1])).T,
                       'x1x2x3+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1),
                                                 xIn[:, 2].reshape(xIn.shape[0], 1),
                                                 xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x1x2x3+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1),
                                                   xIn[:, 1].reshape(xIn.shape[0], 1),
                                                   xIn[:, 2].reshape(xIn.shape[0], 1),
                                                   xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x1x2x3+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1),
                                                     xIn[:, 1].reshape(xIn.shape[0], 1),
                                                     xIn[:, 2].reshape(xIn.shape[0], 1),
                                                     xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x1x2x4+t1': np.vstack((xIn[:, 0], xIn[:, 1], xIn[:, 3], xOut[:, i - 1])).T,
                       'x1x2x4+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 1].reshape(xIn.shape[0], 1),
                                                 xIn[:, 3].reshape(xIn.shape[0], 1),
                                                 xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x1x2x4+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1),
                                                   xIn[:, 1].reshape(xIn.shape[0], 1),
                                                   xIn[:, 3].reshape(xIn.shape[0], 1),
                                                   xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x1x2x4+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1),
                                                     xIn[:, 1].reshape(xIn.shape[0], 1),
                                                     xIn[:, 3].reshape(xIn.shape[0], 1),
                                                     xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x1x3x4+t1': np.vstack((xIn[:, 0], xIn[:, 2], xIn[:, 3], xOut[:, i - 1])).T,
                       'x1x3x4+t1t2': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                                 xIn[:, 3].reshape(xIn.shape[0], 1),
                                                 xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x1x3x4+t1t2t3': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1),
                                                   xIn[:, 2].reshape(xIn.shape[0], 1),
                                                   xIn[:, 3].reshape(xIn.shape[0], 1),
                                                   xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x1x3x4+t1t2t3t4': np.hstack((xIn[:, 0].reshape(xIn.shape[0], 1),
                                                     xIn[:, 2].reshape(xIn.shape[0], 1),
                                                     xIn[:, 3].reshape(xIn.shape[0], 1),
                                                     xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x2x3x4+t1': np.vstack((xIn[:, 1], xIn[:, 2], xIn[:, 3], xOut[:, i - 1])).T,
                       'x2x3x4+t1t2': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1), xIn[:, 2].reshape(xIn.shape[0], 1),
                                                 xIn[:, 3].reshape(xIn.shape[0], 1),
                                                 xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x2x3x4+t1t2t3': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1),
                                                   xIn[:, 2].reshape(xIn.shape[0], 1),
                                                   xIn[:, 3].reshape(xIn.shape[0], 1),
                                                   xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x2x3x4+t1t2t3t4': np.hstack((xIn[:, 1].reshape(xIn.shape[0], 1),
                                                     xIn[:, 2].reshape(xIn.shape[0], 1),
                                                     xIn[:, 3].reshape(xIn.shape[0], 1),
                                                     xOut[:, i - 4:i].reshape(xOut.shape[0], 4))),
                       'x1x2x3x4+t1': np.append(xIn[:, :], xOut[:, i - 1].reshape(xOut.shape[0], 1), axis=1),
                       'x1x2x3x4+t1t2': np.hstack((xIn, xOut[:, i - 2:i].reshape(xOut.shape[0], 2))),
                       'x1x2x3x4+t1t2t3': np.hstack((xIn, xOut[:, i - 3:i].reshape(xOut.shape[0], 3))),
                       'x1x2x3x4+t1t2t3t4': np.hstack((xIn, xOut[:, i - 4:i].reshape(xOut.shape[0], 4)))}
        return inputs_dict
