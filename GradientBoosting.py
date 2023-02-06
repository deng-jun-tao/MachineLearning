import numpy as np
from sklearn.tree import DecisionTreeRegressor  # base learner
# from DecisionTree import DecisionTreeRegressor # my learner already runable but too slow
from LossFunction import SoftmaxLoss, SquareLoss, HuberLoss, get_n_dummies

'''
my DecisionTree base learner is already usable, but it's too slow.
I will use my base learner to replace sklearn learner after optimize my decision tree.
'''

# 1 Regress
class GradientBoostingRegressor():
    def __init__(self,
                 loss='squared_error',
                 n_estimators=100,
                 lr=0.1,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_depth=3,
                 ccp_alpha=0):
        self.n_estimators = n_estimators
        self.lr = lr
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        if loss == 'squared_error':
            self.loss = SquareLoss()
        elif 'huber' in loss:
            self.loss = HuberLoss(int(loss[5:]))
        else:
            self.loss = loss
        self.trees = []
        for i in range(n_estimators):
            self.trees.append(DecisionTreeRegressor(min_samples_leaf=self.min_samples_leaf,
                                                    min_samples_split=self.min_samples_split,
                                                    max_depth=self.max_depth,
                                                    ccp_alpha=self.ccp_alpha))

    def fit(self, X, y, verbose=False):
        self.trees[0].fit(X, y)
        y_pred = self.trees[0].predict(X)
        gradient = self.loss.gradient(y, y_pred)
        for i in range(1, self.n_estimators):
            self.trees[i].fit(X, -gradient)
            y_pred += self.trees[i].predict(X)*self.lr
            gradient = self.loss.gradient(y, y_pred)
            # print info
            if verbose and i % verbose == 0:
                print(
                    f'Turn:{i}\t\tAvg Loss:{self.loss.loss(y,y_pred).mean()}')
        return self

    def predict(self, X):
        y_pred = self.trees[0].predict(X)
        for i in range(1, self.n_estimators):
            y_pred += self.trees[i].predict(X)*self.lr
        return y_pred


# 2 Classifier
# need to train n*m trees for m-classification
class GradientBoostingClassifier():
    def __init__(self,
                 n_estimators=100,
                 n_class=-1,
                 lr=0.1,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_depth=3,
                 ccp_alpha=0.0):
        self.n_estimators = n_estimators
        self.n_class = n_class
        self.lr = lr
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.loss = SoftmaxLoss()
        self.trees = []

    def _init_trees(self):
        for i in range(self.n_estimators):
            self.trees.append([DecisionTreeRegressor(min_samples_leaf=self.min_samples_leaf,
                                                     min_samples_split=self.min_samples_split,
                                                     max_depth=self.max_depth,
                                                     ccp_alpha=self.ccp_alpha)
                              for j in range(self.n_class)])
        return self

    def fit(self, X, y, verbose=False):
        if self.n_class == -1:
            # this n_class logic should be impoved, now it only support y are 0,1,2,...,n
            self.n_class = y.max()+1
        self._init_trees()
        y_pred = np.zeros((len(y), self.n_class))
        # actually this is not a gradient, the fisrt set of trees are fitted on y_true not gradient, just for convience's sake
        gradient = -get_n_dummies(y, n_class=self.n_class)
        # fit
        for num in range(self.n_estimators):
            for i in range(self.n_class):
                self.trees[num][i].fit(X, -gradient[:, i])
                y_pred[:, i] += self.trees[num][i].predict(X)
            gradient = self.loss.gradient(y, y_pred)
            # print loss
            if verbose and num % verbose == 0:
                print(
                    f'Turn:{num}\t\tAvg Loss:{self.loss.loss(y,y_pred).mean()}')

        return self

    def predict_proba(self, X):
        y_pred = np.zeros((len(X), self.n_class))
        for num in range(self.n_estimators):
            for i in range(self.n_class):
                y_pred[:, i] += self.trees[num][i].predict(X)
        return self.loss.softmax(y_pred)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)