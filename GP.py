# modified from https://github.com/belakaria/MESMOC

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel,Matern

class GaussianProcess():
    def __init__(self, dim):
        self.dim = dim
        self.kernel =  RBF(length_scale=0.5, length_scale_bounds=(1e-3, 1e2))
        self.model = GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=5)

        self.X = np.array([])
        self.y = np.array([])

    def fitModel(self):
        self.model.fit(self.X, self.y)

    def addSamples(self, x, y):
        if len(self.X) == 0:
            self.X = np.array([x])
            self.y = np.array([y])
        else:
            self.X = np.vstack((self.X, [x]))
            self.y = np.vstack((self.y, [y]))

    def getPrediction(self, x):
        mean, std = self.model.predict(x, return_std=True)
        if len(mean.shape) == 2:
            mean = mean[:,0]
        return mean, std

    def check_if_x_exists(self, inp_x):
        return (any((inp_x == x).all() for x in self.X))

class MultiObjsGaussianProcess_Ind():
    def __init__(self, dim, nb_objs):
        self.dim = dim
        self.nb_objs = nb_objs
        self.GPs = []
        for _ in range(nb_objs):
            self.GPs.append(GaussianProcess(dim))
    
    def check_if_x_exists(self, inp_x, obj_ind):
        return self.GPs[obj_ind].check_if_x_exists(inp_x)
    
    def addSamples(self, x, y):
        assert(len(x) == self.dim)
        assert(len(y) == self.nb_objs)
        for i in range(self.nb_objs):
            self.GPs[i].addSamples(x, y[i])

    def fitModel(self):
        for i in range(self.nb_objs):
            self.GPs[i].fitModel()

    def getPrediction(self, x):
        means = []
        stds = []
        for i in range(self.nb_objs):
            mean, std = self.GPs[i].getPrediction(x)
            means.append(np.expand_dims(mean, axis=1))
            stds.append(np.expand_dims(std, axis=1))

        return np.concatenate(means, axis=1), np.concatenate(stds, axis=1), 









