# all of these functions are under the assumption that
# 1. there are m different samples as input
# 2. the input is an np.array

import numpy as np
import pandas as pd


def get_n_dummies(y, n_class=-1):
    '''
    eg:
    [0,3,1,0]->
    [
        [1,0,0,0],
        [0,0,0,1],
        [0,1,0,0],
        [1,0,0,0]
    ]
    pd.get_dummies can't do this, so I write this function
    '''
    if n_class == -1:
        n_class = max(y)+1
    res = np.zeros((len(y), n_class))
    tmp = pd.get_dummies(y)
    res[:, tmp.columns] = tmp
    return res


class SquareLoss():
    def loss(self, y_true, y_pred):
        res = (y_true-y_pred)**2
        return res.sum()

    def gradient(self, y_true, y_pred):
        return (y_pred-y_true)


class HuberLoss():
    def __init__(self, delta=1):
        self.delta = delta

    def loss(self, y_true, y_pred):
        diff = y_pred-y_true
        case_A = np.abs(diff) < self.delta
        tmp = case_A*0.5*diff**2 + \
            (~case_A)*(self.delta*np.abs(diff)-0.5*self.delta**2)
        return tmp.sum()

    def gradient(self, y_true, y_pred):
        diff = y_pred-y_true
        case_A = np.abs(diff) < self.delta
        return case_A*diff+(~case_A)*self.delta*np.sign(diff)


class SoftmaxLoss():
    '''
    n_class = n ,
    samples size = m
    >>> y_true.shape = (m,)
    >>> y_pred.shape = (m,n)
    '''

    def softmax(self, y_pred):
        res = np.e**y_pred/(np.e**y_pred).sum(axis=1).reshape(-1, 1)
        return res

    def loss(self, y_true, y_pred):
        y = get_n_dummies(y_true, len(y_pred[0]))
        proba = self.softmax(y_pred)
        return -(y*np.log(proba)).sum(axis=1)

    def gradient(self, y_true, y_pred):
        # this gradient result is rather simple
        # for more detail, see https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        y = get_n_dummies(y_true)
        proba = self.softmax(y_pred)
        return -(y-proba)