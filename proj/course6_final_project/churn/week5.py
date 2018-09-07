import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split, learning_curve
from sklearn.metrics import roc_auc_score
from ..churn.pipeline import (load_data,
                              load_test_data,
                              get_pipeline,
                              filter_outliers,
                              undersample)


def run_week():

    def q1():
        X_train, X_hold, X_test, y_train, y_hold = load_data()

        X_train = X_train.iloc[:10000, :]
        y_train = y_train.iloc[:10000]

        # preprocess
        X_train, y_train = filter_outliers(X_train, y_train, X_train.columns[:190], 0.01)

        model = XGBClassifier(learning_rate=0.1, n_estimators=120, random_state=9, n_jobs=-1)
        pipeline = get_pipeline(model)

        train_sizes, train_scores, test_scores = learning_curve(pipeline,
                                                                X_train, y_train,
                                                                train_sizes=np.arange(0.1, 1.0, 0.5),
                                                                cv=3, scoring='roc_auc')

        plt.grid(True)
        plt.plot(train_sizes, train_scores.mean(axis = 1), 'g-', marker='o', label='train')
        plt.plot(train_sizes, test_scores.mean(axis = 1), 'r-', marker='o', label='test')
        plt.ylim((0.0, 1.05))
        plt.legend(loc='lower right')
        plt.show()
    #q1()

    def q21():
        X_train, X_hold, X_test, y_train, y_hold = load_data()

        X_train = X_train.iloc[:10000, :]
        y_train = y_train.iloc[:10000]

        # preprocess
        X_train, y_train = filter_outliers(X_train, y_train, X_train.columns[:190], 0.01)

        cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=9)

        ws = [0.5, 1, 10]
        for w in ws:
            weights = np.ones_like(y_train)
            weights[y_train==1] = w

            model = XGBClassifier(learning_rate=0.1, n_estimators=120, random_state=9, n_jobs=-1)
            pipeline = get_pipeline(model)

            score = cross_val_score(pipeline,
                                    X_train, y_train,
                                    scoring='roc_auc', cv=cv,
                                    fit_params={ 'model__sample_weight': weights})
            print(w, score.mean())
    #q21()

    def q22():

        X_train, X_hold, X_test, y_train, y_hold = load_data()

        X_train = X_train.iloc[:10000, :]
        y_train = y_train.iloc[:10000]

        # preprocess
        X_train, y_train = filter_outliers(X_train, y_train, X_train.columns[:190], 0.01)

        cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=9)

        u_coeffs = [0.9, 0.5, 0.1]
        for coeff in u_coeffs:
            X_utrain, y_utrain = undersample(X_train, y_train, coeff)

            model = XGBClassifier(learning_rate=0.1, n_estimators=120, random_state=9, n_jobs=-1)
            pipeline = get_pipeline(model)

            score = cross_val_score(pipeline,
                                    X_utrain, y_utrain,
                                    scoring='roc_auc', cv=cv)
            print(coeff, score.mean())
    #q22()

    def q3():
        X_train, X_hold, X_test, y_train, y_hold = load_data()

        X_train = X_train.iloc[:10000, :]
        y_train = y_train.iloc[:10000]

        # preprocess
        X_train, y_train = filter_outliers(X_train, y_train, X_train.columns[:190], 0.01)

        cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=9)

        missings = ['mean', 'median', 'most_frequent']
        for missing in missings:
            model = XGBClassifier(learning_rate=0.1, n_estimators=120, random_state=9, n_jobs=-1)
            pipeline = get_pipeline(model, missing=missing)

            score = cross_val_score(pipeline,
                                    X_train, y_train,
                                    scoring='roc_auc', cv=cv)
            print(missing, score.mean())
    #q3()

    def q4():
        X_train, X_hold, X_test, y_train, y_hold = load_data()

        # preprocess
        X_train, y_train = filter_outliers(X_train, y_train, X_train.columns[:190], 0.01)

        X_train = X_train[:7000]
        y_train = y_train[:7000]

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=9)

        encoders = ['frequency', 'mean_target', 'dummy']
        for encoder in encoders:
            model = XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=9, n_jobs=-1)
            pipeline = get_pipeline(model, encoder=encoder)
            score = cross_val_score(pipeline,
                                    X_train, y_train,
                                    scoring='roc_auc', cv=cv)
            print(encoder, score.mean())
    #q4()

    def q5():
        X_train, X_hold, X_test, y_train, y_hold = load_data()

        # preprocess
        X_train, y_train = filter_outliers(X_train, y_train, X_train.columns[:190], 0.01)

        X_train = X_train[:10000]
        y_train = y_train[:10000]

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=9)

        selectors = [
            ('correlation', { 'n_top': 30 }),
            ('correlation', { 'n_top': 50 }),
            ('correlation', { 'n_top': 100 }),
            ('lasso_svc',   { 'C': 0.01 }),
            ('lasso_svc',   { 'C': 1 }),
            ('lasso_svc',   { 'C': 10 }),
           ]
        for selector, sel_params in selectors:
            model = XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=9, n_jobs=-1)
            pipeline = get_pipeline(model, selector=selector, sel_params=sel_params)
            score = cross_val_score(pipeline,
                                    X_train, y_train,
                                    scoring='roc_auc', cv=cv)
            print(selector, sel_params, score.mean())
        return
    #q5()

    def q6():
        X_train, X_hold, X_test, y_train, y_hold = load_data()

        # preprocess
        X_train, y_train = filter_outliers(X_train, y_train, X_train.columns[:190], 0.01)

        X_train = X_train[:10000]
        y_train = y_train[:10000]

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=9)

        params = {
            'model__n_estimators': [ 50, 100, 200, 300],
            'model__learning_rate': [ 0.01, 0.1, 1 ],
            'model__max_depth': [ 3, 5, 7 ]
            }
        model = XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=9, n_jobs=-1)
        pipeline = get_pipeline(model, selector='correlation', sel_params={ 'n_top': 30 })
        grid = GridSearchCV(pipeline, params, scoring='roc_auc', cv=cv)
        grid.fit(X_train, y_train)
        print(grid.best_params_)
        print(grid.best_score_)

        best_model = grid.best_estimator_

        return
    #q6()

    def q7():
        cols = best_model.named_steps['feature_selection'].new_cols
        num_col_names = X_train.columns.values[:190]
        cat_col_names = best_model.named_steps['processing'] \
                              .transformer_list[1][1] \
                              .named_steps['encoding'] \
                              .dummy_columns
        col_names = np.concatenate((num_col_names, cat_col_names))[cols]

        importances = best_model.named_steps['model'].feature_importances_
        feature_stats = dict(zip(col_names, importances))

        feature_stats = list(sorted(feature_stats.items(), key=lambda kv: kv[1], reverse=True))
        best_features = feature_stats[:5]
        worst_features = feature_stats[-5:]

        print('Лучшие признаки:', best_features)
        print('Худшие признаки:', worst_features)
        return
    q7()



    return
