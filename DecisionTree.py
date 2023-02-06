import pandas as pd
import numpy as np

# TODO 
# 0 regressor add feature: ccp_alpha
# 1 optimize program speed (pd.value_counts() is very slow, especially H(y) or gini(y) is called many times)
# 2 merge regressor and classifor?
# 3 add more impurity/loss criterion

def H(y):
    '''
    calculate the entropy for a array y (catagory)
    '''
    count_arr = pd.value_counts(y,sort=False).to_numpy()
    p_arr = count_arr/count_arr.sum()
    return np.sum(-p_arr*np.log(p_arr))

def gini(y):
    '''
    calculate the gini index for a array y (catagory)
    '''
    count_arr = pd.value_counts(y,sort=False).to_numpy()
    p_arr = count_arr/count_arr.sum()
    return np.sum(p_arr*(1-p_arr))


def MSE(y):
    '''
    calculate the MSE for an array y (continuous)
    '''
    return np.sum((y-y.mean())**2)


class TreeNode:
    def __init__(self, res=-1, col=-1, col_val=None, class_counts=[], level=0, left=None, right=None):
        self.col = col
        self.col_val = col_val
        self.res = res
        self.class_counts = class_counts
        self.level = level  # not 100% reliable, call root.set_child_level()
        self.left:TreeNode = left
        self.right:TreeNode = right
        return

    def __str__(self) -> str:
        # return '\n'.join(['{0}: {1}'.format(item[0], item[1]) for item in self.__dict__.items()])
        return '\n'.join([f'{item[0]}:{item[1]}' if item[0] not in {'left','right'} else''\
            for item in self.__dict__.items()])

    def is_leaf(self):
        return (not self.left) and (not self.right)
    
    def get_max_depth(self):
        return self.level if self.is_leaf() else max(self.left.get_max_depth(),self.right.get_max_depth())

    def set_child_level(self):
        if self.left:
            self.left.level = self.level + 1
            self.left.set_child_level()
        if self.right:
            self.right.level = self.level + 1
            self.right.set_child_level()
        return

    def classify_data(self, x):
        if self.col == -1:
            return self.res
        if x[self.col] < self.col_val:
            if self.left:
                return self.left.classify_data(x)
            return self.res
        else:
            if self.right:
                return self.right.classify_data(x)
            return self.res

    def _preorder_transversal(self,func):
        func(self)
        if self.left:
            self._preorder_transversal(self.left)
        if self.right:
            self._preorder_transversal(self.right)
        return

        

class DecisionTreeClassifier:
    def __init__(self,
                 criterion='gini',
                 min_impurity_decrease=0.0001,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_depth=100,
                 ccp_alpha=0):
        self.criterion = criterion
        if self.criterion == 'gini':
            self.impurity_f = gini
        elif self.criterion=='entropy':
            self.impurity_f = H
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.trees:TreeNode = None
        return

    def _find_best_divide(self, x, y):
        '''
        _build_tree() need this
        '''
        f = self.impurity_f
        unique_val = pd.value_counts(x).index.sort_values()
        n = len(unique_val)
        cut_arr = np.zeros(n)
        cut_arr[0] = unique_val[0]
        for i in range(1, n):
            cut_arr[i] = (unique_val[i-1] + unique_val[i])/2
        best_impurity = f(y)
        best_cut = unique_val[0]
        for cut in cut_arr:
            # check self.min_samples_leaf
            if len(y[x >= cut])<self.min_samples_leaf:
                continue
            if len(y[x < cut])<self.min_samples_leaf:
                continue
            impurity = f(y[x >= cut])+f(y[x < cut])
            if impurity < best_impurity:
                best_impurity = impurity
                best_cut = cut
        return best_impurity, best_cut

    def _count_y(self, y):
        '''
        _build_tree() need this
        '''
        res = np.zeros(self.n_class, dtype=int)
        tmp = pd.value_counts(y,sort=False)
        res[tmp.index] = tmp
        return res

    def _build_tree(self, X, y, node):
        '''
        make inplace change
        (also return node result)
        '''
        if len(X) == 0:
            return None
        class_counts = self._count_y(y)
        node.res = np.argmax(class_counts)
        node.class_counts = class_counts
        '''# I know normally this is not necessary, because _build_tree() will return root node
        # But to record the level(depth), I need to set the root before hand
        if self.trees is None:
            self.trees = node'''
        # check min_samples_split
        if len(X) < self.min_samples_split:
            return node
        # check max_depth
        self.trees.set_child_level()
        if node.level >= self.max_depth:
            return node
        # calculate information entropy for every feature
        impurity = np.zeros(self.n_feature)
        impurity_new = np.zeros(self.n_feature)
        cut_val = np.zeros(self.n_feature)
        for i in range(self.n_feature):
            impurity[i] = self.impurity_f(y)
            impurity_new[i], cut_val[i] = self._find_best_divide(X[:, i], y)
        # find max entropy decrease
        col = np.argmax(impurity-impurity_new)
        cut = cut_val[col]
        # check min_impurity_decrease
        if impurity[col]-impurity_new[col] < self.min_impurity_decrease:
            return node
        # check min_samples_leaf
        is_left = X[:, col] < cut
        if is_left.sum() < self.min_samples_leaf or (~is_left).sum() < self.min_samples_leaf:
            return node
        # finish check, start split
        # dfs
        else:
            # set cut condition
            node.col = col
            node.col_val = cut
            # set cut result (left&right)
            # is_left = X[:, col] < cut (already defined above)
            node.left = TreeNode()
            node.right = TreeNode()
            self._build_tree(X[is_left], y[is_left],node.left)
            self._build_tree(X[~is_left], y[~is_left],node.right)
        return node


    def _compute_R_for_all_leaf(self, node: TreeNode, R_sum=0, leaf_cnt=0):
        '''
        _ccp_alpha_for_a_node() need this function
        '''
        if node.is_leaf():
            # R_sum += n_wrong/n_all (for each leaf)
            R_sum += (sum(node.class_counts) - max(node.class_counts))/self._n_sample
            leaf_cnt += 1
            return R_sum, leaf_cnt
        else:
            R_sum, leaf_cnt = self._compute_R_for_all_leaf(node.left,R_sum,leaf_cnt)
            R_sum, leaf_cnt = self._compute_R_for_all_leaf(node.right,R_sum,leaf_cnt)
        return R_sum, leaf_cnt

    def _ccp_alpha_for_a_node(self, node: TreeNode):
        '''
        _post_prune() need this function
        _post_prune() do:
            post pruning (Cost Complexity Pruning)
            check self.ccp_alpha
        '''
        if node.is_leaf():
            # don't prune leaf
            # just return alpha_max=1 ?
            return 1
        R_sum, leaf_cnt = self._compute_R_for_all_leaf(node)
        res = 0
        res += (sum(node.class_counts)-max(node.class_counts))/self._n_sample
        res -= R_sum
        res /= leaf_cnt-1
        return res

    def _post_prune(self, node):
        '''
        post pruning (Cost Complexity Pruning)
        check self.ccp_alpha
        '''
        if not node:
            return
        if self._ccp_alpha_for_a_node(node) < self.ccp_alpha:
            node.left = None
            node.right = None
            return
        self._post_prune(node.left)
        self._post_prune(node.right)
        return

    def fit(self, X, y):
        self.trees = TreeNode()
        self._n_sample = X.shape[0]
        self.n_feature = X.shape[1]
        self.n_class = len(pd.value_counts(y))
        # start fit: build tree
        self._build_tree(X, y, self.trees)
        # start post prune
        if self.ccp_alpha>0:
            self._post_prune(self.trees)
        return self

    def predict(self, X):
        n = len(X)
        y_pred = np.zeros(n)
        for i in range(n):
            y_pred[i] = self.trees.classify_data(X[i])
        return y_pred



class DecisionTreeRegressor:
    def __init__(self,
                 criterion='MSE',
                 min_impurity_decrease=0.0001,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_depth=100,
                 ccp_alpha=0.0001):
        self.criterion = criterion
        if self.criterion=='MSE':
            self.impurity_f = MSE
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha #TODO
        self.trees:TreeNode = None
        return

    def _find_best_divide(self, x, y):
        '''
        _build_tree() need this
        '''
        f = self.impurity_f
        unique_val = pd.value_counts(x,sort=False).index.sort_values()
        n = len(unique_val)
        cut_arr = np.zeros(n)
        cut_arr[0] = unique_val[0]
        for i in range(1, n):
            cut_arr[i] = (unique_val[i-1] + unique_val[i])/2
        best_impurity = f(y)
        best_cut = unique_val[0]
        for cut in cut_arr:
            # check self.min_samples_leaf
            if len(y[x >= cut])<self.min_samples_leaf:
                continue
            if len(y[x < cut])<self.min_samples_leaf:
                continue
            impurity = f(y[x >= cut])+f(y[x < cut])
            if impurity < best_impurity:
                best_impurity = impurity
                best_cut = cut
        return best_impurity, best_cut


    def _build_tree(self, X, y, node):
        '''
        make inplace change
        (also return node result)
        '''
        if len(X) == 0:
            return None
        node.res = np.mean(y) #TODO: this is different
        # check min_samples_split
        if len(X) < self.min_samples_split:
            return node
        # check max_depth
        self.trees.set_child_level()
        if node.level >= self.max_depth:
            return node
        # calculate loss(MSE, etc.) for every feature
        impurity = np.zeros(self.n_feature)
        impurity_new = np.zeros(self.n_feature)
        cut_val = np.zeros(self.n_feature)
        for i in range(self.n_feature):
            impurity[i] = self.impurity_f(y)
            impurity_new[i], cut_val[i] = self._find_best_divide(X[:, i], y)
        # find max entropy decrease
        col = np.argmax(impurity-impurity_new)
        cut = cut_val[col]
        # check min_impurity_decrease #TODO: this is different
        if impurity[col]-impurity_new[col] < self.min_impurity_decrease*len(y):
            return node
        # check min_samples_leaf
        is_left = X[:, col] < cut
        if is_left.sum() < self.min_samples_leaf or (~is_left).sum() < self.min_samples_leaf:
            return node
        # finish check, start split
        # dfs
        else:
            # set cut condition
            node.col = col
            node.col_val = cut
            # set cut result (left&right)
            # is_left = X[:, col] < cut (already defined above)
            node.left = TreeNode()
            node.right = TreeNode()
            self._build_tree(X[is_left], y[is_left],node.left)
            self._build_tree(X[~is_left], y[~is_left],node.right)
        return node


    def _compute_R_for_all_leaf(self, node: TreeNode, R_sum=0, leaf_cnt=0):
        '''
        _ccp_alpha_for_a_node() need this function
        '''
        if node.is_leaf():
            # R_sum += n_wrong/n_all (for each leaf)
            R_sum += (sum(node.class_counts) - max(node.class_counts))/self._n_sample
            leaf_cnt += 1
            return R_sum, leaf_cnt
        else:
            R_sum, leaf_cnt = self._compute_R_for_all_leaf(node.left,R_sum,leaf_cnt)
            R_sum, leaf_cnt = self._compute_R_for_all_leaf(node.right,R_sum,leaf_cnt)
        return R_sum, leaf_cnt

    def _ccp_alpha_for_a_node(self, node: TreeNode):
        '''
        _post_prune() need this function
        _post_prune() do:
            post pruning (Cost Complexity Pruning)
            check self.ccp_alpha
        '''
        if node.is_leaf():
            # don't prune leaf
            # just return alpha_max=1 ?
            return 1
        R_sum, leaf_cnt = self._compute_R_for_all_leaf(node)
        res = 0
        res += (sum(node.class_counts)-max(node.class_counts))/self._n_sample
        res -= R_sum
        res /= leaf_cnt-1
        return res

    def _post_prune(self, node):
        '''
        post pruning (Cost Complexity Pruning)
        check self.ccp_alpha
        '''
        if not node:
            return
        if self._ccp_alpha_for_a_node(node) < self.ccp_alpha:
            node.left = None
            node.right = None
            return
        self._post_prune(node.left)
        self._post_prune(node.right)
        return

    def fit(self, X, y):
        self.trees = TreeNode()
        self._n_sample = X.shape[0]
        self.n_feature = X.shape[1]
        # start fit: build tree
        self._build_tree(X, y, self.trees)
        # start post prune
        '''if self.ccp_alpha>0:
            self._post_prune(self.trees)'''
        return self

    def predict(self, X):
        n = len(X)
        y_pred = np.zeros(n)
        for i in range(n):
            y_pred[i] = self.trees.classify_data(X[i])
        return y_pred