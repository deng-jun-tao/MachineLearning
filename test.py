import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split


def test_classification(model):
    '''bunch = datasets.load_wine()  # Classification
    X = pd.DataFrame(bunch['data'], columns=bunch['feature_names']).to_numpy()
    y = pd.Series(bunch['target']).to_numpy()'''
    X,y = datasets.make_classification(n_samples=1000,random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('\nacc:', sklearn.metrics.accuracy_score(y_test, y_pred))


def test_regression(model):
    X, y = datasets.make_regression(
        n_samples=1000, n_features=12, tail_strength=0.1,random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=442)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('R^2:', sklearn.metrics.r2_score(y_test, y_pred))


def main():
    # 1 Decision Tree - classify
    from DecisionTree import DecisionTreeClassifier
    model = DecisionTreeClassifier(
        max_depth=10, min_impurity_decrease=0.001, ccp_alpha=0.001)
    test_classification(model)

    # 2 GBDT - classify
    '''from GradientBoosting import GradientBoostingClassifier
    model = GradientBoostingClassifier(ccp_alpha=0.001)
    test_classification(model)'''

    # 3 Decision Tree - regress
    '''from DecisionTree import DecisionTreeRegressor
    model = DecisionTreeRegressor()
    test_regression(model)'''

    # 4 GBDT - regress
    '''from GradientBoosting import GradientBoostingRegressor
    model = GradientBoostingRegressor()
    test_regression(model)'''


if __name__ == '__main__':
    main()


'''
# 分析性能 命令行运行 python -m cProfile -s cumulative -o tmp.stats test.py
# 然后使用下方代码查看函数性能报告
import pstats
p = pstats.Stats("tmp.stats")
p.sort_stats("cumulative") 
p.print_stats()  # 展示函数调用时间分析结果
p.print_callers()  # 展示函数的被调用情况
p.print_callees()  # 展示函数的调用情况
'''