import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from calendar import monthrange


def header():
    return 'WEEK 1: Business Applications';


def run():

    #quiz1()
    peer_review()

    return


def quiz1():

    df = pd.read_csv(utils.PATH.COURSE_FILE(5, 'monthly-milk-production.csv', 'week1'),
                     sep=';')

    print(df.head())
    print(df.columns)

    #plt.plot(df['month'], df['milk'])
    #plt.show()

    res = adfuller(df['milk'])
    print(res)

    def average(row):
        date = pd.to_datetime(row['month'], dayfirst=True)
        days = monthrange(date.year, date.month)[1]
        return row['milk'] / days

    df['daily'] = df.apply(average, axis=1)

    #plt.plot(df['month'], df['daily'], 'b')
    #plt.show()

    print(df.head())
    print(df['daily'].values.sum())

    df['daily_diff1']  = df['daily'] - df['daily'].shift(1)
    df['daily_diff12'] = df['daily'] - df['daily'].shift(12)
    df['daily_diff12_1'] = df['daily_diff12'] - df['daily_diff12'].shift(1)
    df.dropna(inplace=True)


    res = adfuller(df['daily_diff1'])
    print(res)
    res = adfuller(df['daily_diff12'])
    print(res)
    res = adfuller(df['daily_diff12_1'])
    print(res)

    plot_acf(df['daily_diff12_1'].values.squeeze(), lags=50)
    plt.show()

    plot_pacf(df['daily_diff12_1'].values.squeeze(), lags=50)
    plt.show()

    return


def peer_review():

    def invboxcox(y,lmbda):
        if lmbda == 0:
            return(np.exp(y))
        else:
            return(np.exp(np.log(lmbda*y+1)/lmbda))

    df = pd.read_csv(utils.PATH.COURSE_FILE(5, 'WAG_C_M.csv', 'week1'),
                     sep=';',
                     index_col=['month'],
                     parse_dates=['month'],
                     dayfirst=True)
    print(df.head())

    plt.plot(df.index, df['WAG_C_M'])
    plt.show()

    # Проверка стационарности и STL-декомпозиция ряда:
    plt.figure(figsize=(15,10))
    seasonal_decompose(df['WAG_C_M']).plot()
    plt.show()
    print("Критерий Дики-Фуллера: p=%f" % adfuller(df['WAG_C_M'])[1])

    # Сделаем преобразование Бокса-Кокса для стабилизации дисперсии
    df['WAG_C_M_box'], lmbda = stats.boxcox(df['WAG_C_M'])
    plt.figure(figsize=(15,7))
    plt.ylabel(u'Transformed WAG')
    df['WAG_C_M_box'].plot()
    plt.show()
    print("Оптимальный параметр преобразования Бокса-Кокса: %f" % lmbda)
    print("Критерий Дики-Фуллера: p=%f" % adfuller(df['WAG_C_M_box'])[1])

    # Стационарность
    df['WAG_C_M_box_diff'] = df['WAG_C_M_box'] - df['WAG_C_M_box'].shift(12)
    plt.figure(figsize=(15,10))
    seasonal_decompose(df['WAG_C_M_box_diff'][12:]).plot()
    plt.show()
    print("Критерий Дики-Фуллера: p=%f" % adfuller(df['WAG_C_M_box_diff'][12:])[1])

    # Попробуем добавить ещё обычное дифференцирование
    df['WAG_C_M_box_diff2'] = df['WAG_C_M_box_diff'] - df['WAG_C_M_box_diff'].shift(1)
    plt.figure(figsize=(15,10))
    seasonal_decompose(df['WAG_C_M_box_diff2'][13:]).plot()
    plt.show()
    print("Критерий Дики-Фуллера: p=%f" % adfuller(df['WAG_C_M_box_diff2'][13:])[1])

    # Посмотрим на ACF и PACF полученного ряда:

    ## Далее - в wag.ipynb

    return