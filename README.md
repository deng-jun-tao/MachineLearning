> 中文/English

## 一些介绍
这个仓库是我使用Python实现一些机器学习模型的代码。主要基于numpy和pandas库，尽可能实现sklearn中的模型。

## 已完成的部分

#### 基本的损失函数

- 平方损失
- Huber损失
- Softmax损失

#### 决策树

- 分类
- 回归

包括 预剪枝（不纯度限制、分裂限制、深度限制等）、CCP后剪枝(ccp_alpha) 等相关功能。
回归树暂时不是特别完善，有一定的修改空间，而且在大样本上速度相对较慢，需要后续优化。

#### GBDT
- 分类
- 回归

分类器训练了n_class*n_estimator个基学习器，使用一对多（One vs Rest）的方式进行多分类。

目前暂时使用了sklearn.tree中的决策树作为基学习器。我写的决策树可以正常替换、完成对应的功能，但速度太慢了。后续完成性能分析、优化后会进行替换。

## Some Notes

This repository aims to implement some ML models using Python.

I want to use this as my code training, and as an record of my ML learning.

## Completed section

#### Basic loss function

- squared loss
- Huber loss
- Softmax loss

#### Decision Tree

- Classification
- Regression

Including pre-pruning (impurity limit, split limit, depth limit, etc.), CCP post-pruning (ccp_alpha) and other related functions.
The regression tree is not particularly perfect for the time being, there is a certain room for modification, and the speed is relatively slow on large samples, and subsequent optimization is required.

#### GBDT
- Classification
- Regression

The classifier trains n_class*n_estimator base learners, and uses a one-to-many (One vs Rest) method for multi-classification.

Currently, the decision tree in sklearn.tree is temporarily used as the base learner. The decision tree I wrote can be replaced normally and complete the corresponding functions, but the speed is too slow. Subsequent performance analysis and optimization will be replaced.

