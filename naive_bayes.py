# -*- coding: utf-8 -*-
# author: ykomashiro@gmail.com
import numpy as np


class Naive_Bayes():
    def __init__(self, X_tr, y_tr):
        self.X = X_tr
        self.y = y_tr
        self.cls = np.unique(y_tr)  # type of class
        self.Pc = (np.bincount(y_tr))/len(y_tr)
        self.mu, self.sigma = self._Gassian_Param()

    def _Gassian_Param(self):
        D = self.X.shape[1]
        mu = np.empty((len(self.cls), D))
        sigma = np.empty((len(self.cls), D))
        for i in range(len(self.cls)):
            X_batch = self.X[self.y == i]
            mu[i, :] = np.mean(X_batch, axis=1)
            sigma[i, :] = np.var(X_batch, ddof=1, axis=1)
        return mu, sigma

    def _pdf(self, x):
        eps = 1e-5
        denominator = np.sqrt(2*np.pi+eps)*self.sigma
        numerator = np.exp(-((x-self.mu)**2)/(2*(self.sigma+eps)**2))
        return np.sum(np.log(numerator/denominator), axis=0)

    def train(self):
        pass

    def predict(self, X_te, y_te=None):
        pass
