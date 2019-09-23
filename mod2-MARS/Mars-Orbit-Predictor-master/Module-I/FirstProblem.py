import numpy as np
from scipy.stats.mstats import gmean
from scipy.optimize import minimize
import math


class FirstProblem:
    def __init__(self, params, alphas, betas):
        self._alphas = alphas
        self._betas = betas
        self._params = params

    def get_angular_position_and_radius_of_mars(self, x, y, alpha, beta):
        c1 = np.sin(beta - y)
        c2 = x * np.sin(alpha - y)
        c3 = np.cos(beta - alpha)
        radius = np.sqrt((2*c3*c2*c1 + c1*c1 + c2*c2)/(1 - c3*c3))
        phi = alpha + np.arcsin(c2/radius)
        return phi, radius

    def func_to_minimize_arithmetic_geometric_mean(self, params, args):
        radius_list = []
        alphas = args[0]
        betas = args[1]
        assert len(alphas) == len(betas)
        for i in range(len(alphas)):
            _, radius = self.get_angular_position_and_radius_of_mars(params[0], params[1], alphas[i], betas[i])
            radius_list.append(radius)

        arithmetic_mean = np.mean(radius_list)
        geometric_mean = gmean(radius_list)
        return arithmetic_mean - geometric_mean

    def func_to_minimize_sample_variance(self, params, args):
        radius_list = []
        alphas = args[0]
        betas = args[1]
        assert len(alphas) == len(betas)
        for i in range(len(alphas)):
            _, radius = self.get_angular_position_and_radius_of_mars(params[0], params[1], alphas[i], betas[i])
            radius_list.append(radius)

        sample_variance = np.var(radius_list)
        return sample_variance

    def minimize_loss_function(self, func_type):
        if func_type == 1:
            params = minimize(self.func_to_minimize_arithmetic_geometric_mean, self._params,
                              args=[self._alphas, self._betas],
                              bounds=[(0, None), (None, None)])
        else:
            params = minimize(self.func_to_minimize_sample_variance, self._params,
                              args=[self._alphas, self._betas],
                              bounds=[(0, None), (None, None)])
        return params['x'][0], params['x'][1]
