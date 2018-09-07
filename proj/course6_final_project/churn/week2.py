import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from ..churn.utils import get_data


def run_week():

    X_train, X_test, y_train, y_test = get_data(test_size=0.2)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # y-stratification
    print(y_train.mean())
    print(y_test.mean())

    # Q1
    holdout_df = pd.concat([X_test, y_test], axis=1)
    holdout_df.to_csv(utils.PATH.STORE_FOR(6, 'churn_holdout.csv', 'churn'))
    print(holdout_df.shape)


    # Q2
    # Подумайте и предложите несколько способов (не менее 3х) обработки категориальных признаков, для того, чтобы их можно было использовать при построении модели.

    #1. One-hot-encoding - классический способ: создаем столько новых фич, сколько различных значений у категориального признака, далее ставим 1 для соответствующей новой фичи
    #2. Label encoding - можем просто для каждой переменной занумеровать значения в порядке их появления 1, 2, 3,... и далее закодировать значения этими числами. Это может быть не очень хорошо, так как мы вводим естественный порядок в категориальные переменные, а его там может не быть
    #3. Target encoding - кодируем каждое значение категории средним значением целевой переменной на ней
    #4. Frequency encoding - кодируем каждое значение категории частотой его появления в данной переменной
    #5. Комбинация разных категориальных фич - конкатенация всевозможных значений двух фич и далее One-hot-encoding
    #6. Hashing Trick

    #Q3
    #Подумайте, с помощью какой метрики качества лучше всего оценивать качество будущей модели, какой будет ключевая метрика качества?

    #Наиболее естесственная бизнес-метрика здесь - accuracy: число правильных ответов.
    #Для оценки качества моделей в задачах бинарной классификации удобно использовать метрику ROC-AUC, #так она
    #- зачастую оказывается наиболее естественной
    #- хорошо коррелирует с accuracy
    #- нечувствительна к дисбалансу классов
    #- реализована практически во всех фреймворках

    #Q4
    #Какие вспомогательные метрики качества имеет смысл считать, чтобы более полно оценить качество модели, понять, где её слабые стороны и что нужно улучшать?

    # Вывод моделей, скорее всего, будет вероятностно-подобным. Имеет смысл попробовать log-loss и даже MSE, MAE метрики
    # Precision, recall и F-мера

    #Q5
    #Подберите оптимальную стратегию проведения кросс-валидации: решите, на сколько фолдов вы будете делить выборку?
    #Выберите тип разбиения данных (k-fold, stratified k-fold, shuffle split, leave one out).

    #Обычно рекомендуемое оптимальное число фолдов для разбиения - 5-7.
    #Можно использовать 7, если данных не слишком мало и компьютер позволяет. Это наш случай.
    #В качестве типа разбиения данных будем использовать stratified k-fold, так как имеет место дисбаланс классов.
    #Leave one out слишком дорог, так как данных многовато.

    print('-------------')
    df = pd.read_csv(utils.PATH.STORE_FOR(6, 'review1_week2.txt', 'churn'))
    print(df.shape)
    y = df.iloc[:, -1].apply(lambda x: 1 if x==1 else 0)
    print(y.mean())

    print('-------------')
    df = pd.read_csv(utils.PATH.STORE_FOR(6, 'review2_week2.csv', 'churn'))
    print(df.shape)
    y = df.iloc[:, -1].apply(lambda x: 1 if x==1 else 0)
    print(y.mean())

    print('-------------')
    data_df = pd.read_csv(utils.PATH.STORE_FOR(6, 'review3_week2_data.csv', 'churn'))
    label_df = pd.read_csv(utils.PATH.STORE_FOR(6, 'review3_week2_labels.csv', 'churn'))
    print(data_df.shape)
    y = label_df.iloc[:, -1].apply(lambda x: 1 if x==1 else 0)
    print(y.mean())


    return

