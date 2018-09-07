import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report, roc_curve
from ..churn.utils import get_data, get_test_data, get_baseline_pipeline, get_optimal_threshold


def run_week():

    #model = RandomForestClassifier(random_state=9, n_jobs=-1, n_estimators=150)
    #params = { 'model__n_estimators': [100, 150, 200] }

    #model = Ridge(alpha=0.1, random_state=9)

    model = LogisticRegression(C=10, class_weight='balanced', random_state=9)
    params = { 'model__C': [1, 10, 100] }

    run_model(model, params, 5)
    #run_model_no_grid_search(model, 5)

    return


def run_model(model, params, n_folds):
    print('*'*64)
    print(' Running model: {0} '.format(type(model).__name__).center(64, '*'))
    print('*'*64)

    #X_train, X_test, y_train, y_test = get_test_data()
    #num_cnt = 3
    #n_folds = 2

    X_train, X_test, y_train, y_test = get_data(test_size=0.2)
    X_train = X_train.iloc[:10000, :]
    y_train = y_train.iloc[:10000, :].values.ravel()

    pipeline = get_baseline_pipeline(X_train, 190, model)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=9)

    metrics = [ ('precision', precision_score), ('roc_auc', roc_auc_score), ('recall', recall_score), ('f1', f1_score) ]
    for metric, score_func in metrics:
        print('Metric: {0}'.format(metric).center(64))

        grid = GridSearchCV(pipeline, params, scoring=metric, cv=cv, verbose=10)
        grid.fit(X_train, y_train)

        y_pred = grid.predict_proba(X_test)[:, 1]

        fpr, tpr, thr = roc_curve(y_test, y_pred)
        threshold = get_optimal_threshold(fpr, tpr, thr)
        print('optimal threshold:', threshold)
        if metric != 'roc_auc':
            y_pred = y_pred > threshold

        ho_score = score_func(y_test, y_pred)

        print('Metric {0} results:'.format(metric))
        print('CV score:', cv_score)
        print('Holdout score:', ho_score)
        print('Report:')
        print(classification_report(y_test, y_pred>threshold))

    print('\n-------------------------------------\n')

    return


def run_model_no_grid_search(model, n_folds):
    print('*'*64)
    print(' Running model: {0} '.format(type(model).__name__).center(64, '*'))
    print('*'*64)

    X_train, X_test, y_train, y_test = get_data(test_size=0.2)
    X_train = X_train.iloc[:10000, :]
    y_train = y_train.iloc[:10000, :].values.ravel()

    pipeline = get_baseline_pipeline(X_train, 190, model)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=9)

    metrics = [ ('precision', precision_score), ('roc_auc', roc_auc_score), ('recall', recall_score), ('f1', f1_score) ]
    for metric, score_func in metrics:
        print('Metric: {0}'.format(metric).center(64))

        cv_score = cross_val_score(pipeline, X_train, y_train, scoring=metric, cv=cv, verbose=10).mean()

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict_proba(X_test)[:, 1]

        fpr, tpr, thr = roc_curve(y_test, y_pred)
        threshold = get_optimal_threshold(fpr, tpr, thr)
        print('optimal threshold:', threshold)
        if metric != 'roc_auc':
            y_pred = y_pred > threshold

        ho_score = score_func(y_test, y_pred)

        print('Metric {0} results:'.format(metric))
        print('CV score:', cv_score)
        print('Holdout score:', ho_score)
        print('Report:')
        print(classification_report(y_test, y_pred>threshold))

    print('\n-------------------------------------\n')

    return
