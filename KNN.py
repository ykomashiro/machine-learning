# -*- coding: utf-8 -*-
# author: ykomashiro@gmail.com
import numpy as np
import operator


def calEDist(arrA, arrB):
    return np.sum((arrA-arrB)**2)


def getNeighbors(trainSet, cls_true, target, k):
    N = trainSet.shape[0]
    dists = []
    neighbors = np.empty((k,))
    for i in range(N):
        dist = calEDist(trainSet[i], target)
        dists.append((i, dist))
    dists.sort(key=operator.itemgetter(1))
    for j in range(k):
        neighbors[j] = trainSet[dists[j][0]]
    result = cls_true[neighbors]
    cls_pred = np.argmax(np.bincount(result))
    return cls_pred


def predict(k, trainSet, cls_true, testSet, cls_test=None):
    N = testSet.shape[0]
    cls_pred = []
    acc = None
    for i in range(N):
        pred = getNeighbors(trainSet, cls_true, testSet[i, :], k)
        cls_pred.append(pred)
    if cls_test is not None:
        acc = np.mean(np.array(cls_pred) == cls_test)
    return np.array(cls_pred), acc
