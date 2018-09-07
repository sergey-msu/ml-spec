import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from ..churn.utils import (get_data,
                           get_test_data,
                           get_baseline_pipeline,
                           get_optimal_threshold,
                           preprocess_and_get_pipeline,
                           point_biserial_corr)


def run_week():

    train_df = pd.read_csv(utils.PATH.COURSE_FILE(6, 'orange_small_churn_train_data.csv', 'churn'), index_col='ID')
    test_df  = pd.read_csv(utils.PATH.COURSE_FILE(6, 'orange_small_churn_test_data.csv', 'churn'), index_col='ID')
    y = train_df['labels'].apply(lambda x: 1 if x==1 else 0)
    X = train_df.drop(['labels'], axis=1)

    X_train, X_hold, \
    y_train, y_hold = train_test_split(X, y,
                                       test_size=0.2,
                                       random_state=9,
                                       shuffle=True,
                                       stratify=y)
    X_test = test_df

    model = XGBClassifier(learning_rate=0.1, n_estimators=100, n_jobs=-1)
    params = {
        'model__learning_rate': [0.1],
        'model__n_estimators':  [120] }

    X_train, y_train, pipeline = preprocess_and_get_pipeline(X_train, y_train, model)

    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=9)

    grid = GridSearchCV(pipeline, params, scoring='roc_auc', cv=cv, verbose=10)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_score_)

    model = grid.best_estimator_

    # score on holdout set
    y_pred = model.predict_proba(X_hold)[:, 1]
    score = roc_auc_score(y_hold, y_pred)
    print('Holdout score:', score)

    # submit test

    model.fit(X, y)
    y_pred = model.predict_proba(X_test)[:, 1]

    res_df = pd.DataFrame(y_pred, columns=['result'])
    res_df.index.name = 'ID'
    res_df.to_csv(utils.PATH.STORE_FOR(6, 'churn_res.csv', 'churn'), sep=',')

    return
