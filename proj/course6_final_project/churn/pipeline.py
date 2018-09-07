import math
import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, Imputer, StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.svm import LinearSVC
from sklearn.metrics import matthews_corrcoef


def load_data():
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


def load_test_data():
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


def get_pipeline(model,
                  missing='mean',
                  encoder='dummy', enc_params=None,
                  selector=None,   sel_params=None):

    # choose encoder

    enc_params = {} if enc_params is None else enc_params
    if encoder == 'dummy':
        encoder = DummyEncoder(**enc_params)
    elif encoder == 'mean_target':
        encoder = MeanTargetEncoder(**enc_params)
    elif encoder == 'frequency':
        encoder = FrequencyEncoder(**enc_params)
    else:
        encoder = NopeTransformer()

    # choose selector

    sel_params = {} if sel_params is None else sel_params
    if selector == 'lasso_svc':
        selector = LassoSelector(**sel_params)
    elif selector == 'correlation':
        selector = CorrelationSelector(**sel_params)
    else:
        selector = NopeTransformer()

    # construct pipeline

    pipeline = Pipeline(steps=[
            # processing
            ('processing', FeatureUnion([

                # numeric features
                ('numeric', Pipeline(steps=[
                    ('selecting',      FunctionTransformer(lambda data: data.iloc[:, :190], validate=False)),
                    ('float_nan_mean', Imputer(strategy=missing)),
                    ('scaling',        StandardScaler())
                ])),

                # categorical features
                ('categorical',   Pipeline(steps=[
                    ('selecting', FunctionTransformer(lambda data: data.iloc[:, 190:], validate=False)),
                    ('encoding',  encoder)
                ]))
            ])),

            # feature selection
            ('feature_selection', selector),

            # model
            ('model', model)
        ])

    return pipeline


class DummyEncoder(BaseEstimator, TransformerMixin):
    '''
    Encodes categorical features as one-hot variables with max_categories restriction
    '''
    def __init__(self, columns=None, max_categories=200):
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


class CorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_top=100):
        self.n_top = n_top


    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1. filter out const columns
        vc = X.apply(lambda col: len(col.value_counts()))
        all_cols = vc[vc > 1].index.values

        # 2. correlation feature selection
        num_corrs = X[all_cols].apply(lambda col: correlation(col.values, y), axis=0)
        top_corrs = num_corrs.abs().sort_values(ascending=False)[:self.n_top]

        self.new_cols = sorted(top_corrs.index)

        return self


    def transform(self, X):
        check_is_fitted(self, ['new_cols'])

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return X.loc[:, self.new_cols]


class LassoSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=None, C=0.1):
        self.threshold = threshold
        self.C = C
        return


    def fit(self, X, y=None):
        model = LinearSVC(C=self.C, penalty='l1', dual=False)
        model.fit(X, y)
        self.selector = SelectFromModel(model, prefit=True, threshold=self.threshold)
        return self


    def transform(self, X):
        check_is_fitted(self, ['selector'])
        return self.selector.transform(X)


class NopeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def filter_outliers(X, y, cols, alpha):
    print('filtering outliers...')
    for col in cols:
        var = X[col]
        var_churn = var[y==1]
        var_loyal = var[y==0]

        outliers = len(X)
        condition = None
        col_a = alpha

        while outliers > 200:
            churn_min, churn_max = var_churn.quantile([col_a, 1 - col_a])
            loyal_min, loyal_max = var_loyal.quantile([col_a, 1 - col_a])

            condition = var.isnull() | \
                        ((y==1) & (churn_min <= var) & (var <= churn_max)) | \
                        ((y==0) & (loyal_min <= var) & (var <= loyal_max))

            outliers = len(X) - len(X[condition])
            col_a /= 2

        if condition is not None:
            X = X[condition]
            y = y[condition]
    print('finished: ', len(X))

    return X, y


def undersample(X, y, coeff):

    np.random.seed(9)

    churn = X[y==1].index
    loyal = X[y==0].index.values
    np.random.shuffle(loyal)
    u_loyal = loyal[: int(coeff*len(loyal))]

    u_ids = list(churn) + list(u_loyal)

    return X.ix[u_ids, :], y.ix[u_ids]


def correlation(x, y):
    if set(np.unique(x)) == { 0.0, 1.0 }:
        return matthews_corrcoef(x, y)
    else:
        return point_biserial_corr(x, y)


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


class DebugChecker(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
