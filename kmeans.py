# -*- coding: utf-8 -*-
# copyright: ykomashiro@gmail.com
import numpy as np


class KMeans():
    def __init__(self, k=3, max_iter=500):
        '''
        initialize some paramters
            :param k: the clusters we want to divide.
            :param max_iter: the max iteration if not convergence.
        '''
        self._k = k
        self._max_iter = max_iter
        self._clusterAssment = None
        self._labels = None
        self._sse = None

    def _calEDist(self, arrA, arrB):
        """
        calculate the distance between two difference vector.
            :param arrA: an array of shape (N,1).
            :param arrB: an array of shape (N,1).
        """
        return np.sqrt(np.sum((arrA-arrB)**2))

    def _randCenter(self, data, k):
        """
        docstring here 
            :param data: the input data of shape (N, D).
            :param k: the clusters we want to divide.
        """
        D = data.shape[1]
        centroids = np.empty((k, D))  # cluters
        for j in range(D):
            minJ = min(data[:, j])
            rangeJ = float(max(data[:, j]-minJ))
            centroids[:, j] = (minJ+rangeJ*np.random.rand(k, 1)).flatten()
        return centroids

    def fit(self, data):
        """
        main algrithrom execute.
            :param data: the input data of shape (N, D).
        """
        N = data.shape[0]
        # to store the data of cluster.
        self._clusterAssment = np.zeros((N, 2))
        self._centroids = self._randCenter(data, self._k)
        clusterChange = True
        for itr in range(self._max_iter):
            print(itr)
            clusterChange = False
            for i in range(N):
                minDist = np.inf
                minIndex = -1
                for j in range(self._k):
                    arrA = self._centroids[j, :]
                    arrB = data[i, :]
                    distJI = self._calEDist(arrA, arrB)
                    if distJI < minDist:
                        # update
                        minDist = distJI
                        minIndex = j
                if self._clusterAssment[i, 0] != minIndex or self._clusterAssment[i, 1] > minDist**2:
                    clusterChange = True
                    self._clusterAssment[i, :] = minIndex, minDist**2
            if not clusterChange:
                break
            for i in range(self._k):
                index_all = self._clusterAssment[:, 0]
                value = np.nonzero(index_all == i)
                ptsInClust = data[value[0]]
                self._centroids[i, :] = np.mean(ptsInClust, axis=0)
        self._labels = self._clusterAssment[:, 0]
        self._sse = sum(self._clusterAssment[:, 1])

    def predict(self, X):
        m = X.shape[0]
        preds = np.empty((m,))
        for i in range(m):
            minDist = np.inf
            for j in range(self._k):
                distJI = self._calEDist(self._centroids[j, :], X[i, :])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds


def data_load(path):
    fn = np.loadtxt(path, delimiter=",")
    data = fn[:, 0:-1].astype(np.float)
    label = fn[:, -1].astype(np.int16)
    label[label == -1] = 0
    return data, label


if __name__ == "__main__":
    data, labels = data_load(
        r'project\machine learning\data\breast-cancer.csv')
    classifier = KMeans(2)
    classifier.fit(data)
    a = classifier.predict(data)
    print(np.mean(a == labels))
