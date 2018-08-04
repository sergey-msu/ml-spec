import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
from xgboost import XGBRegressor
from collections import Counter
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import mean_squared_error


def header():
    return 'WEEK 4: Decision Tree, Forest and Gradient Boosting';


def run():

    #test()
    #homework_1()
    homework_2()

    return


def test():
    classification_problem = datasets.make_classification(n_features = 2, n_informative = 2,
                                                          n_classes = 3, n_redundant=0,
                                                          n_clusters_per_class=1, random_state=3)

    colors = ListedColormap(['red', 'blue', 'yellow'])
    light_colors = ListedColormap(['lightcoral', 'lightblue', 'lightyellow'])

    #plt.figure(figsize=(8,6))
    #plt.scatter(list(map(lambda x: x[0], classification_problem[0])),
    #            list(map(lambda x: x[1], classification_problem[0])),
    #            c=classification_problem[1],
    #            cmap=colors,
    #            s=100)
    #plt.show()

    X_train, X_test, \
    y_train, y_test = train_test_split(classification_problem[0],
                                       classification_problem[1],
                                       test_size = 0.3,
                                       random_state = 1)

    alg = DecisionTreeClassifier(random_state=1)
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(acc)

    alg = DecisionTreeClassifier(random_state = 1, max_depth = 1)
    alg.fit(X_train, y_train)

    #plot_decision_surface(alg, X_train, y_train, X_test, y_test, colors, light_colors)
    #plt.show()

    alg = DecisionTreeClassifier(random_state = 1, max_depth = 2)
    alg.fit(X_train, y_train)
    #plot_decision_surface(alg, X_train, y_train, X_test, y_test, colors, light_colors)
    #plt.show()

    alg = DecisionTreeClassifier(random_state = 1, max_depth = 3)
    alg.fit(X_train, y_train)
    #plot_decision_surface(alg, X_train, y_train, X_test, y_test, colors, light_colors)
    #plt.show()

    alg = DecisionTreeClassifier(random_state = 1)
    alg.fit(X_train, y_train)
    #plot_decision_surface(alg, X_train, y_train, X_test, y_test, colors, light_colors)
    #plt.show()

    alg = DecisionTreeClassifier(random_state = 1, min_samples_leaf = 3)
    alg.fit(X_train, y_train)
    plot_decision_surface(alg, X_train, y_train, X_test, y_test, colors, light_colors)
    plt.show()

    return


def get_meshgrid(data, step=.05, border=.5):
    x_min, x_max = data[:, 0].min() - border, data[:, 0].max() + border
    y_min, y_max = data[:, 1].min() - border, data[:, 1].max() + border
    return np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))


def plot_decision_surface(estimator,
                          train_data, train_labels, test_data, test_labels,
                          colors, light_colors):
    #set figure size
    plt.figure(figsize = (16, 6))

    #plot decision surface on the train data
    plt.subplot(1,2,1)
    xx, yy = get_meshgrid(train_data)
    mesh_predictions = np.array(estimator.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap = light_colors)
    plt.scatter(train_data[:, 0], train_data[:, 1], c = train_labels, s = 100, cmap = colors)
    plt.title('Train data, accuracy={:.2f}'.format(metrics.accuracy_score(train_labels, estimator.predict(train_data))))

    #plot decision surface on the test data
    plt.subplot(1,2,2)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap = light_colors)
    plt.scatter(test_data[:, 0], test_data[:, 1], c = test_labels, s = 100, cmap = colors)
    plt.title('Test data, accuracy={:.2f}'.format(metrics.accuracy_score(test_labels, estimator.predict(test_data))))

    return


def homework_1():

    data = datasets.load_digits()
    X = data.data
    y = data.target

    # Q1
    alg = DecisionTreeClassifier()
    score = cross_val_score(alg, X, y, cv=10, n_jobs=-1)
    print(score, score.mean())

    # Q2
    bag = BaggingClassifier(alg, n_estimators=100)
    score = cross_val_score(bag, X, y, cv=10, n_jobs=-1)
    print(score, score.mean())

    # Q3
    n_feat = int(math.sqrt(X.shape[1]))
    bag = BaggingClassifier(alg, n_estimators=100, max_features=n_feat)
    score = cross_val_score(bag, X, y, cv=10, n_jobs=-1)
    print(score, score.mean())

    # Q4
    alg = DecisionTreeClassifier(max_features=n_feat)
    bag = BaggingClassifier(alg, n_estimators=100)
    score = cross_val_score(bag, X, y, cv=10, n_jobs=-1)
    print(score, score.mean())

    # Q5
    print('----------------------')
    n_estimators_list = [5, 10, 15, 100, 200, 500, 1000]
    for n_estimators in n_estimators_list:
        alg = RandomForestClassifier(n_estimators=n_estimators, max_features=n_feat)
        score = cross_val_score(alg, X, y, cv=10, n_jobs=-1)
        print(n_estimators, score.mean())

    print('----------------------')
    n_feats = [5, 10, 40, 50, 60]
    for n_feat in n_feats:
        alg = RandomForestClassifier(n_estimators=100, max_features=n_feat)
        score = cross_val_score(alg, X, y, cv=10, n_jobs=-1)
        print(n_feat, score.mean())

    print('----------------------')
    max_depths = [5, 6, 7, 8, 15]
    for max_depth in max_depths:
        alg = RandomForestClassifier(n_estimators=100, max_depth=max_depth)
        score = cross_val_score(alg, X, y, cv=10, n_jobs=-1)
        print(max_depth, score.mean())

    return


def homework_2():

    data = datasets.load_boston()
    X = data.data
    y = data.target

    l = len(y)
    test_size = int(0.25*l)

    X_train, X_test = X[:-test_size, :], X[-test_size:, :]
    y_train, y_test = y[:-test_size], y[-test_size:]

    # Q2,3

    base_algs = [] # for DecisionTreeRegressors
    coeffs = []    # their weights
    n_estimators = 50
    max_depth = 5
    random_state = 42
    coeff = 0.9

    def gbm_predict(X):
        return [sum([coeff * alg.predict([x])[0] for alg, coeff in zip(base_algs, coeffs)]) for x in X]

    for i in range(n_estimators):
        alg = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        xgb_train = [0]*(l - test_size) if i==0 else y_train - gbm_predict(X_train)
        alg.fit(X_train, xgb_train)
        base_algs.append(alg)
        coeffs.append(coeff/(1.0 + i))

    y_pred = gbm_predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    print(math.sqrt(score))

    # Q4

    for n_estimators in range(1, 100000, 10000):
        alg = XGBRegressor(n_estimators=n_estimators, n_jobs=-1)
        alg.fit(X_train, y_train)
        y_pred = alg.predict(X_test)
        score = mean_squared_error(y_test, y_pred)
        print(n_estimators, math.sqrt(score))

    for max_depth in range(1, 1000, 100):
        alg = XGBRegressor(n_estimators=50, max_depth=max_depth, n_jobs=-1)
        alg.fit(X_train, y_train)
        y_pred = alg.predict(X_test)
        score = mean_squared_error(y_test, y_pred)
        print(max_depth, math.sqrt(score))

    # Q5

    alg = LinearRegression()
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    print(math.sqrt(score))

    return