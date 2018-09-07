import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import seaborn as sns
import itertools as it
from matplotlib.lines import Line2D
from pandas.plotting import scatter_matrix
from matplotlib.colors import ListedColormap


def run_week():

    data_df  = pd.read_csv(utils.PATH.COURSE_FILE(6, 'orange_small_churn_data.txt', 'churn'))
    label_df = pd.read_csv(utils.PATH.COURSE_FILE(6, 'orange_small_churn_labels.txt', 'churn'), header=None, names=['target'])

    label_df['target'] = (label_df['target'] + 1) // 2
    print(data_df.head())

    churn_df = data_df[label_df['target'] == 1]
    loyal_df = data_df[label_df['target'] == 0]

    print('churn ratio:', len(churn_df)/len(data_df), ',  loyal ratio:', len(loyal_df)/len(data_df))


    num_corrs = data_df.iloc[:, :190].apply(lambda col: point_biserial_corr(col.values, label_df['target'].values), axis=0)
    print(num_corrs.head())


    top20_corrs = num_corrs.abs().sort_values(ascending=False)[:20]
    print(top20_corrs)


    plot_scatter(data_df, label_df, top20_corrs.index, 4)


    random.seed(9)
    n_feats = 10
    idxs = random.sample(range(190), n_feats)

    cols = data_df.columns[idxs]
    plot_scatter(data_df, label_df, cols, 4, figsize=(16, 10))
    plot_hist(data_df, label_df, cols, 4)


    low10_corrs = num_corrs.sort_values(ascending=True)[:10]
    plot_scatter(data_df, label_df, low10_corrs.index, 4, figsize=(16, 10))
    plot_hist(data_df, label_df, low10_corrs.index, 4)


    plot_cat_classes(data_df, label_df, data_df.columns.values[190:230], 5)


    ## 8. Проанализируйте полученные результаты:
    ##
    ## - Какие выводы вы можете сделать? Заметили ли вы какие-нибудь интересные закономерности?
    ## - На основании полученных изображений и таблиц, предположите, какие переменные окажут наибольшее влияние (вклад) в модель?
    ## - Какие переменные на ваш взгляд окажутся наименее полезными, шумовыми?
    ## - Как вы думаете, окажется ли отбор признаков полезным для построения модели?
    ##
    ## Выводы:
    ##

    return


def point_biserial_corr(x, y):
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    p = y.mean()
    q = 1 - p
    ex = x.mean()
    sx = x.std(ddof=0)

    px = x[y==1]
    nx = x[y==0]

    return (px.mean() - nx.mean())/sx*math.sqrt(p*q)


def plot_scatter(data_df, label_df, cols, n_cols, figsize=(16, 80)):
    colors = ListedColormap(['blue', 'red'])
    plot_data = []

    for col_pair in it.combinations(cols, 2):
        col1, col2 = col_pair
        cols_df = pd.concat([data_df[[col1, col2]], label_df], axis=1)
        cols_df.dropna(inplace=True)
        if len(cols_df) > 0:
            plot_data.append((col1, col2, cols_df))

    n_rows = math.ceil(len(plot_data)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    legend_lines = [Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='red', lw=4)]
    plt.figlegend(legend_lines, ['Loyal', 'Churn'], loc = 'upper center', ncol=n_cols, labelspacing=0. )

    for idx, p in enumerate(plot_data):
        col1, col2, cols_df = p
        i = idx//n_cols
        j = idx%n_cols
        ax = axes[i, j]
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.scatter(cols_df[col1], cols_df[col2], c=cols_df['target'], s=13, cmap = colors, alpha=0.5)

    plt.show()


def plot_hist(data_df, label_df, cols, n_cols, figsize=(16, 10)):
    n_rows = math.ceil(len(cols)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    legend_lines = [Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='red', lw=4)]
    plt.figlegend(legend_lines, ['Loyal', 'Churn'], loc = 'upper center', ncol=n_cols, labelspacing=0. )

    for idx, col in enumerate(cols):
        i = idx//n_cols
        j = idx%n_cols
        ax = axes[i, j]
        ax.text(.5,.9, col, horizontalalignment='center', transform=ax.transAxes)

        x = data_df[col].dropna()
        x[label_df['target']==0].hist(ax = ax, bins=50, color='blue', alpha=0.7, density=True)
        x[label_df['target']==1].hist(ax = ax, bins=50, color='red', alpha=0.7, density=True)

    plt.show()


def plot_cat_classes(data_df, label_df, cols, n_cols, cat_limit=200, figsize=(16, 20)):
    plot_data = []

    for col in cols:
        x = data_df[col].dropna()
        x_loyal = x[label_df['target']==0].value_counts()
        x_churn = x[label_df['target']==1].value_counts()

        if len(x_loyal)>cat_limit or len(x_churn)>cat_limit:
            continue

        plot_data.append((col, x_loyal,x_churn))

    n_rows = math.ceil(len(plot_data)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    legend_lines = [Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='red', lw=4)]
    plt.figlegend(legend_lines, ['Loyal', 'Churn'], loc = 'upper center', ncol=n_cols, labelspacing=0. )

    for idx, p in enumerate(plot_data):
        col, x_loyal, x_churn = p
        i = idx//n_cols
        j = idx%n_cols
        ax = axes[i, j]
        ax.text(.5, .9, col, horizontalalignment='center', transform=ax.transAxes)

        if len(x_loyal) > 0:
            x_loyal.plot(ax = ax, kind='bar', color='blue', alpha=0.7)
        if len(x_churn) > 0:
            x_churn.plot(ax = ax, kind='bar', color='red', alpha=0.7)

    plt.show()
