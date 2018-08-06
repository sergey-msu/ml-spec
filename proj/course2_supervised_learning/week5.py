import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
from collections import Counter
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


def header():
    return 'WEEK 5: Neural Networsk, Bayesian approach, KNN and SVM';


def run():

    #homework_nn()
    #homework_nb()
    homework_metric()

    return


def homework_nn():

    df = pd.read_csv(utils.PATH.COURSE_FILE(2, 'week5//winequality-red.csv'), sep=';')

    X = df.drop(['quality'], axis=1)
    y = df['quality'].apply(lambda x: 5 if x<5 else (7 if x>7 else x))

    X_train, X_test, \
    y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # NO PYBRAIN :(

    return


def homework_nb():

    data_digits = datasets.load_digits()
    X_digits = data_digits.data
    y_digits = data_digits.target

    data_cancer = datasets.load_breast_cancer()
    X_cancer = data_cancer.data
    y_cancer = data_cancer.target

    print(X_cancer)

    alg_bern = BernoulliNB()
    alg_mult = MultinomialNB()
    alg_gaus = GaussianNB()

    score_bern_digits = cross_val_score(alg_bern, X_digits, y_digits, cv=10).mean()
    score_mult_digits = cross_val_score(alg_mult, X_digits, y_digits, cv=10).mean()
    score_gaus_digits = cross_val_score(alg_gaus, X_digits, y_digits, cv=10).mean()
    print('----- Digits -----')
    print('Bernoulli', score_bern_digits)
    print('Multinomial', score_mult_digits)
    print('Gaussian', score_gaus_digits)

    alg_bern = BernoulliNB()
    alg_mult = MultinomialNB()
    alg_gaus = GaussianNB()

    score_bern_cancer = cross_val_score(alg_bern, X_cancer, y_cancer, cv=10).mean()
    score_mult_cancer = cross_val_score(alg_mult, X_cancer, y_cancer, cv=10).mean()
    score_gaus_cancer = cross_val_score(alg_gaus, X_cancer, y_cancer, cv=10).mean()
    print('----- Brest Cancer -----')
    print('Bernoulli', score_bern_cancer)
    print('Multinomial', score_mult_cancer)
    print('Gaussian', score_gaus_cancer)

    return


def homework_metric():

    data = datasets.load_digits()
    X = data.data
    y = data.target

    train_size = int(0.75*len(y))
    X_train, X_test = X[:train_size, :], X[train_size:, :]
    y_train, y_test = y[:train_size], y[train_size:]

    # Q1

    n_neighbors = 1
    def predict_knn(x, k):
        idxs = np.argpartition(norm(X_train - x, axis=1), k)[:k]
        class_cnts = Counter(y_train[idxs])
        return class_cnts.most_common(1)[0][0]

    y_pred = []
    for x in X_test:
        y_pred.append(predict_knn(x, 1))
    score = 1 - accuracy_score(y_test, y_pred)
    print(score)

    # Q2

    alg = RandomForestClassifier(n_estimators=1000)
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_test)
    score = 1 - accuracy_score(y_test, y_pred)
    print(score)

    return
