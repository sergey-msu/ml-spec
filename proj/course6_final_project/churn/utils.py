import os
import math
import numpy as np
import pandas as pd
import utils
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, Imputer, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
import xgboost as xgb


def get_data(test_size=None):

    X = pd.read_csv(utils.PATH.COURSE_FILE(6, 'orange_small_churn_data.txt', 'churn'))
    y = pd.read_csv(utils.PATH.COURSE_FILE(6, 'orange_small_churn_labels.txt', 'churn'), header=None, names=['target'])
    y['target'] = y['target'].apply(lambda x: 1 if x==1 else 0)

    if test_size is None:
        return X, y

    return train_test_split(X, y,
                            test_size=test_size,
                            random_state=9,
                            shuffle=True,
                            stratify=y)


def get_prod_data():
    train_df = pd.read_csv(utils.PATH.COURSE_FILE(6, 'orange_small_churn_train_data.csv', 'churn'), index_col='ID')
    test_df  = pd.read_csv(utils.PATH.COURSE_FILE(6, 'orange_small_churn_test_data.csv', 'churn'), index_col='ID')
    X = train_df.drop(['labels'], axis=1)
    y = train_df['labels'].apply(lambda x: 1 if x==1 else 0)

    X_train, X_hold, \
    y_train, y_hold = train_test_split(X, y,
                                       test_size=0.2,
                                       random_state=9,
                                       shuffle=True,
                                       stratify=y)
    X_test = test_df

    return X_train, X_hold, X_test, y_train, y_hold


def get_test_data():
    X_train = pd.DataFrame({
        'Col1': [1.0, None, 3.0, 4.0],
        'Col2': [None, None, None, None],
        'Col3': [1.1, 2.1, None, 4.1],
        'Col4': ['A', 'B', None, 'A'],
        'Col5': [None, None, None, None],
        'Col6': ['A', None, None, 'B']
        })
    y_train = pd.DataFrame({ 'target': [1, 0, 0, 1] })
    X_test = pd.DataFrame({
        'Col1': [1.0, None],
        'Col2': [None, 2.0],
        'Col3': [None, None],
        'Col4': ['A', 'C'],
        'Col5': ['B', None],
        'Col6': [None, 'B']
        })
    y_test = pd.DataFrame({ 'target': [1, 0] })

    return X_train, X_test, y_train, y_test


def get_baseline_pipeline(X, num_cnt, alg):

    # get non-NaN columns
    vc = X.apply(lambda col: len(col.value_counts()))
    vc = vc[vc > 0]
    all_cols = X.columns.values
    num_cols = set(all_cols[:num_cnt])
    cat_cols = set(all_cols[num_cnt:])
    non_nan_all_cols = set(vc.index.values)
    non_nan_num_cols = sorted(list(non_nan_all_cols.intersection(num_cols)))
    non_nan_cat_cols = sorted(list(non_nan_all_cols.intersection(cat_cols)))


    pipeline = Pipeline(steps=[
            # get rid of fully NaN columns
            ('filter_out_useless_columns', FunctionTransformer(lambda data: data.loc[:, non_nan_all_cols], validate=False)),

            # processing
            ('processing', FeatureUnion([

                # numeric features
                ('numeric', Pipeline(steps=[
                    ('selecting', FunctionTransformer(lambda data: data.loc[:, non_nan_num_cols], validate=False)),
                    ('float_nan_mean', Imputer(strategy='mean')),
                    ('scaling', StandardScaler())
                ])),

                # categorical features
                ('categorical', Pipeline(steps=[
                    ('selecting', FunctionTransformer(lambda data: data.loc[:, non_nan_cat_cols], validate=False)),
                    ('encoding', DummyEncoder(max_categories=200))
                ]))
            ])),

            #model
            ('model', alg)
        ])

    return pipeline


def preprocess_and_get_pipeline(X_train, y_train, alg):

    # filter out outliers
    X_train, y_train = filter_outliers(X_train, y_train, X_train.columns[:190], 0.01)

    # leave only non-const columns
    non_const_all_cols, non_const_num_cols, non_const_cat_cols = select_features(X_train, y_train)

    pipeline = Pipeline(steps=[
            # get rid of fully NaN columns
            ('filter_out_useless_columns', FunctionTransformer(lambda data: data.loc[:, non_const_all_cols], validate=False)),

            # processing
            ('processing', FeatureUnion([

                # numeric features
                ('numeric', Pipeline(steps=[
                    ('selecting',      FunctionTransformer(lambda data: data.loc[:, non_const_num_cols], validate=False)),
                    ('float_nan_mean', Imputer(strategy='mean')),
                    ('scaling',        StandardScaler())
                ])),

                # categorical features
                ('categorical',   Pipeline(steps=[
                    ('selecting', FunctionTransformer(lambda data: data.loc[:, non_const_cat_cols], validate=False)),
                    ('encoding',  DummyEncoder(max_categories=200))
                ]))
            ])),

            #model
            ('model', alg)
        ])

    return X_train, y_train, pipeline


def get_pipeline(cols, alg, missing='mean', max_categories=200, encoder='dummy'):
    all_cols, num_cols, cat_cols = cols
    if encoder == 'dummy':
        encoder = DummyEncoder(max_categories=max_categories)
    elif encoder == 'mean_target':
        encoder = MeanTargetEncoder()
    elif encoder == 'frequency':
        encoder = FrequencyEncoder()
    else:
        raise ValueError('unknown encoder name:', encoder)


    pipeline = Pipeline(steps=[
            # get rid of useless columns
            ('filter_out_useless_columns', FunctionTransformer(lambda data: data.loc[:, all_cols], validate=False)),

            # processing
            ('processing', FeatureUnion([

                # numeric features
                ('numeric', Pipeline(steps=[
                    ('selecting',      FunctionTransformer(lambda data: data.loc[:, num_cols], validate=False)),
                    ('float_nan_mean', Imputer(strategy=missing)),
                    ('scaling',        StandardScaler())
                ])),

                # categorical features
                ('categorical',   Pipeline(steps=[
                    ('selecting', FunctionTransformer(lambda data: data.loc[:, cat_cols], validate=False)),
                    ('encoding',  encoder)
                ]))
            ])),

            #model
            ('model', alg)
        ])

    return pipeline


class DummyEncoder(BaseEstimator, TransformerMixin):
    '''
    Encodes categorical features as one-hot variables with max_categories restriction
    '''
    def __init__(self, columns=None, max_categories=None):
        self.columns = columns
        self.dummy_columns = None
        self.max_categories = max_categories


    def fit(self, X, y=None, **kwargs):
        self.dummy_columns = None
        return self


    def transform(self, X, y=None, **kwargs):
        if self.max_categories is not None:
            X = X[self.columns] if self.columns is not None else X.copy()
            for col in X.columns:
                top_cats = X[col].value_counts()[:self.max_categories].index.values
                X[col] = X[col].apply(lambda x: x if (x in top_cats or x is None) else 'aggr')

        dummy_df = pd.get_dummies(X, columns=self.columns, sparse=True, dummy_na=True)
        new_cols = dummy_df.columns.values
        if self.dummy_columns is None:
            self.dummy_columns = new_cols
            return dummy_df
        else:
            res_df = pd.DataFrame()
            for col in self.dummy_columns:
                res_df[col] = dummy_df[col] if col in new_cols else np.zeros((len(X),), dtype=int)
        return res_df


class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    '''
    Encodes categorical features by its mean on target variable
    '''
    def __init__(self, columns=None):
        self.columns = columns
        self.dict = None
        return


    def fit(self, X, y=None, **kwargs):
        columns = X.columns if self.columns is None else self.columns
        dict = {}

        X = X.astype(str)

        for col in columns:
            vals = X[col].unique()
            dict[col] = { val: y[X[col] == val].mean() for val in vals }

        self.dict = dict

        return self


    def transform(self, X, y=None, **kwargs):
        check_is_fitted(self, ['dict'])

        X = X.astype(str)
        columns = X.columns if self.columns is None else self.columns

        for col in columns:
            col_dict = self.dict[col]
            X[col] = X[col].apply(lambda x: col_dict.get(x, 0))

        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    '''
    Encodes categorical features by its frequency
    '''
    def __init__(self, columns=None):
        self.columns = columns
        self.dict = None
        return


    def fit(self, X, y=None, **kwargs):
        columns = X.columns if self.columns is None else self.columns
        dict = {}

        X = X.astype(str)
        n = len(X)

        for col in columns:
            vals = X[col].unique()
            dict[col] = { val: (X[col] == val).sum()/n for val in vals }

        self.dict = dict

        return self


    def transform(self, X, y=None, **kwargs):
        check_is_fitted(self, ['dict'])
        X = X.astype(str)
        columns = X.columns if self.columns is None else self.columns

        for col in columns:
            col_dict = self.dict[col]
            X[col] = X[col].apply(lambda x: col_dict.get(x, 0))

        return X


def select_features(X_train, y_train):

    # 1. filter out const columns
    vc = X_train.apply(lambda col: len(col.value_counts()))
    vc = vc[vc > 0]
    all_cols = X_train.columns.values
    num_cols = set(all_cols[:190])
    cat_cols = set(all_cols[190:])
    non_const_all_cols = set(vc.index.values)
    non_const_num_cols = sorted(list(non_const_all_cols.intersection(num_cols)))
    non_const_cat_cols = sorted(list(non_const_all_cols.intersection(cat_cols)))

    # 2. correlation feature selection

    num_corrs = X_train[non_const_num_cols].apply(lambda col: point_biserial_corr(col.values, y_train), axis=0)
    top_corrs = sorted(num_corrs.abs().sort_values(ascending=False)[:100].index)
    all_cols = sorted(list(set(top_corrs).union(non_const_cat_cols)))

    return all_cols, top_corrs, non_const_cat_cols


def filter_outliers(X_train, y_train, cols, alpha):
    print('filtering outliers...')
    for col in cols:
        var = X_train[col]
        var_churn = var[y_train==1]
        var_loyal = var[y_train==0]

        outliers = len(X_train)
        condition = None
        col_a = alpha

        while outliers > 200:
            churn_min, churn_max = var_churn.quantile([col_a, 1 - col_a])
            loyal_min, loyal_max = var_loyal.quantile([col_a, 1 - col_a])

            condition = var.isnull() | \
                        ((y_train==1) & (churn_min <= var) & (var <= churn_max)) | \
                        ((y_train==0) & (loyal_min <= var) & (var <= loyal_max))

            outliers = len(X_train) - len(X_train[condition])
            col_a /= 2

        if condition is not None:
            X_train = X_train[condition]
            y_train = y_train[condition]
    print('finished: ', len(X_train))

    return X_train, y_train


def undersample(X_train, y_train, coeff):

    churn = X_train[y_train==1].index
    loyal = X_train[y_train==0].index.values
    np.random.shuffle(loyal)
    u_loyal = loyal[: int(coeff*len(loyal))]

    u_ids = list(churn) + list(u_loyal)

    return X_train.ix[u_ids, :], y_train.ix[u_ids]



def point_biserial_corr(x, y):
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    p = y.mean()
    q = 1 - p
    ex = x.mean()
    sx = x.std(ddof=0)

    px = x[y==1]
    nx = x[y==0]

    mpx = px.mean() if len(px)>0 else 0
    mnx = nx.mean() if len(nx)>0 else 0

    return (mpx - mnx)/sx*math.sqrt(p*q)


def get_optimal_threshold(fprs, tprs, thrs):
    n = len(fprs)
    dist = 10
    thr_opt = None
    for i in range(n):
        fpr, tpr, thr = fprs[i], tprs[i], thrs[i]
        d = fpr*fpr + (tpr - 1)*(tpr - 1)
        if d < dist:
            thr_opt = thr
            dist = d
    return thr_opt


class DebugChecker(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X








#class FeatureSelector(BaseEstimator, TransformerMixin):
#    def fit(self, X, y=None, **kwargs):
#        # 1. filter out const columns
#        vc = X.apply(lambda col: len(col.value_counts()))
#        vc = vc[vc > 0]
#        all_cols = X.columns.values
#        num_cols = set(all_cols[:190])
#        cat_cols = set(all_cols[190:])
#        non_const_all_cols = set(vc.index.values)
#        non_const_num_cols = sorted(list(non_const_all_cols.intersection(num_cols)))
#        non_const_cat_cols = sorted(list(non_const_all_cols.intersection(cat_cols)))
#
#        print('non-const columns:', len(non_const_all_cols))
#
#        # 2. correlation feature selection
#
#        num_corrs = X[non_const_num_cols].apply(lambda col: point_biserial_corr(col.values, y), axis=0)
#        top_corrs = sorted(num_corrs.abs().sort_values(ascending=False)[:100].index)
#
#        self.all_cols = sorted(list(set(top_corrs).union(non_const_cat_cols)))
#
#        return self
#
#    def transform(self, X, y=None, **kwargs):
#        check_is_fitted(self, ['all_cols', 'num_cols', 'cat_cols'])
#        return X.loc[:, self.all_cols]


