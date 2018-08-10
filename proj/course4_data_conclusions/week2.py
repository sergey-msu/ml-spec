import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from scipy import stats
from scipy.stats import norm, binned_statistic_2d
from statsmodels.stats.weightstats import *
from statsmodels.stats.proportion import proportion_confint, samplesize_confint_proportion
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error


def header():
    return 'WEEK 2: AB-testing and Criterias';


def run():

    #quiz1()
    #quiz2()
    quiz3()

    #test()

    return


def quiz1():

    def q45():
        mu = 9.5
        sigma = 0.4
        n = 160
        mu_obs = 9.57

        Z = (mu_obs - mu)/(sigma/math.sqrt(n))

        pvalue = 2*(1 - stats.norm.cdf(abs(Z)))
        print(Z, pvalue)
        return
    #q45()


    def q67():
        # Q6

        df = pd.read_csv(utils.PATH.COURSE_FILE(4, 'diamonds.txt', 'week2'), sep='\t')
        print(df.head())

        X = df.drop(['price'], axis=1).values
        y = df['price'].values

        X_train, X_test, \
        y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

        model1 = LinearRegression()
        model1.fit(X_train, y_train)
        y_pred1 = model1.predict(X_test)

        sample1 = np.abs(y_test - y_pred1)
        score1 = sample1.sum()/len(y_test)
        print('lin reg score:', score1)

        model2 = RandomForestRegressor(random_state=1)
        model2.fit(X_train, y_train)
        y_pred2 = model2.predict(X_test)

        sample2 = np.abs(y_pred2 - y_test)
        score2 = sample2.sum()/len(y_test)
        print('rand forest', score2)

        # Q7

        res = stats.ttest_ind(sample1, sample2, equal_var = False)
        print(res)

        sample = sample1-sample2
        res = DescrStatsW(sample).tconfint_mean()
        print(res)

        return
    q67()

    return


def quiz2():

    def q3():

        #В одном из выпусков программы "Разрушители легенд" проверялось, действительно ли заразительна зевота.
        #В эксперименте участвовало 50 испытуемых, проходивших собеседование на программу.
        #Каждый из них разговаривал с рекрутером; в конце 34 из 50 бесед рекрутер зевал.
        #Затем испытуемых просили подождать решения рекрутера в соседней пустой комнате.
        #
        #Во время ожидания 10 из 34 испытуемых экспериментальной группы и 4 из 16 испытуемых контрольной начали зевать.
        #Таким образом, разница в доле зевающих людей в этих двух группах составила примерно 4.4%.
        #Ведущие заключили, что миф о заразительности зевоты подтверждён.
        #
        #Можно ли утверждать, что доли зевающих в контрольной и экспериментальной группах отличаются статистически значимо?
        #Посчитайте достигаемый уровень значимости при альтернативе заразительности зевоты, округлите до четырёх знаков после десятичной точки.

        m1 = 10
        n1 = 34
        m2 = 4
        n2 = 16

        conf_interval_1 = proportion_confint(m1, n1, method = 'wilson')
        conf_interval_2 = proportion_confint(m2, n2, method = 'wilson')

        print(conf_interval_1, conf_interval_2)

        diff = proportions_diff_confint_ind(m1, n1, m2, n2)
        print(diff)

        pvalue = proportions_diff_z_test(proportions_diff_z_stat_ind(m1, n1, m2, n2), alternative='greater')
        print('q3', pvalue)

        return
    #q3()

    def q45():

        df = pd.read_csv(utils.PATH.COURSE_FILE(4, 'banknotes.txt', 'week2'), sep='\t')
        print(df.head())

        X = df.drop(['real'], axis=1)
        X123 = X[['X1', 'X2', 'X3']].values
        X456 = X[['X4', 'X5', 'X6']].values
        y = df['real'].values

        X123_train, X123_test, \
        y_train, y_test = train_test_split(X123, y, test_size=50, random_state=1)
        X456_train, X456_test = train_test_split(X456, test_size=50, random_state=1)

        model1 = LogisticRegression()
        model1.fit(X123_train, y_train)
        y123_pred = model1.predict(X123_test)

        model2 = LogisticRegression()
        model2.fit(X456_train, y_train)
        y456_pred = model2.predict(X456_test)

        n = len(y_test)
        n1 = n
        n2 = n
        p1 = accuracy_score(y_test, y123_pred)
        p2 = accuracy_score(y_test, y456_pred)
        print(p1, p2)

        pvalue = proportions_diff_z_test(proportions_diff_z_stat_ind_prob(p2, p1, n2, n1), alternative='two-sided')
        print(pvalue)

        diff = proportions_confint_diff_ind_prob(p1, p2, n1, n2)
        print(diff)

        return
    q45()

    def q67():

        # Q6
        #Ежегодно более 200000 людей по всему миру сдают стандартизированный экзамен GMAT при поступлении на программы MBA.
        #Средний результат составляет 525 баллов, стандартное отклонение — 100 баллов.
        #
        #Сто студентов закончили специальные подготовительные курсы и сдали экзамен. Средний полученный ими балл — 541.4.
        #Проверьте гипотезу о неэффективности программы против односторонней альтернативы о том, что программа работает.
        #Отвергается ли на уровне значимости 0.05 нулевая гипотеза?
        #Введите достигаемый уровень значимости, округлённый до 4 знаков после десятичной точки.

        mu = 525
        sigma = 100
        n = 100
        mu_obs = 541.4

        Z = (mu_obs - mu)/(sigma/math.sqrt(n))

        pvalue = proportions_diff_z_test(Z, 'greater')
        print(Z, pvalue)

        # Q7
        #Оцените теперь эффективность подготовительных курсов, средний балл 100 выпускников которых равен 541.5.
        #Отвергается ли на уровне значимости 0.05 та же самая нулевая гипотеза против той же самой альтернативы?
        #Введите достигаемый уровень значимости, округлённый до 4 знаков после десятичной точки.

        mu = 525
        sigma = 100
        n = 100
        mu_obs = 541.5

        Z = (mu_obs - mu)/(sigma/math.sqrt(n))

        pvalue = proportions_diff_z_test(Z, 'greater')
        print(Z, pvalue)

        return
    q67()

    return


def quiz3():

    def q4():
        #Давайте вернёмся к данным выживаемости пациентов с лейкоцитарной лимфомой из видео про критерий знаков:
        #49, 58, 75, 110, 112, 132, 151, 276, 281, 362*
        #
        #Измерено остаточное время жизни с момента начала наблюдения (в неделях);
        #звёздочка обозначает цензурирование сверху — исследование длилось 7 лет, и остаточное время жизни одного пациента,
        #который дожил до конца наблюдения, неизвестно.
        #
        #Поскольку цензурировано только одно наблюдение, для проверки гипотезы H0:medX=200 на этих данных можно использовать
        #критерий знаковых рангов — можно считать,
        #что время дожития последнего пациента в точности равно 362, на ранг этого наблюдения это никак не повлияет.
        #
        #Критерием знаковых рангов проверьте эту гипотезу против двусторонней альтернативы,
        #введите достигаемый уровень значимости, округлённый до четырёх знаков после десятичной точки.

        x = np.array([49, 58, 75, 110, 112, 132, 151, 276, 281, 362])
        m0 = 200
        res = stats.wilcoxon(x - 200)
        print(res)

        return
    #q4()

    def q5():
        #В ходе исследования влияния лесозаготовки на биоразнообразие лесов острова Борнео
        #собраны данные о количестве видов деревьев в 12 лесах, где вырубка не ведётся:
        #   22, 22, 15, 13, 19, 19, 18, 20, 21, 13, 13, 15,
        #
        #и в 9 лесах, где идёт вырубка:
        #   17, 18, 18, 15, 12, 4, 14, 15, 10.
        #
        #Проверьте гипотезу о равенстве среднего количества видов в двух типах лесов против односторонней альтернативы
        #о снижении биоразнообразия в вырубаемых лесах. Используйте ранговый критерий.
        #Чему равен достигаемый уровень значимости? Округлите до четырёх знаков после десятичной точки.

        x1 = np.array([22, 22, 15, 13, 19, 19, 18, 20, 21, 13, 13, 15])
        x2 = np.array([17, 18, 18, 15, 12, 4, 14, 15, 10])
        res = stats.mannwhitneyu(x1, x2)
        print(res)

        return
    #q5()

    def q67():
        # Q6

        #28 января 1986 года космический шаттл "Челленджер" взорвался при взлёте.
        #Семь астронавтов, находившихся на борту, погибли.
        #В ходе расследования причин катастрофы основной версией была неполадка с резиновыми уплотнительными кольцами в соединении с ракетными ускорителями.
        #Для 23 предшествовавших катастрофе полётов "Челленджера" известны температура воздуха и появление повреждений хотя бы у одного из уплотнительных колец.
        #
        #С помощью бутстрепа постройте 95% доверительный интервал для разности средних температур воздуха при запусках,
        #когда уплотнительные кольца повреждались, и запусках, когда повреждений не было.
        #Чему равна его ближайшая к нулю граница? Округлите до четырёх знаков после запятой.
        #
        #Чтобы получить в точности такой же доверительный интервал, как у нас:
        #
        #установите random seed = 0 перед первым вызовом функции get_bootstrap_samples,
        #один раз сделайте по 1000 псевдовыборок из каждой выборки.

        df = pd.read_csv(utils.PATH.COURSE_FILE(4, 'challenger.txt', 'week2'), sep='\t')
        print(df.head())

        temp_bad = df[df['Incident']==1]['Temperature'].values
        temp_good = df[df['Incident']==0]['Temperature'].values

        np.random.seed(0)

        temp_bad_samples = get_bootstrap_samples(temp_bad, 1000)
        temp_good_samples = get_bootstrap_samples(temp_good, 1000)

        diff_temp_means = temp_bad_samples.mean(axis=1) - temp_good_samples.mean(axis=1)
        res = stat_intervals(diff_temp_means)
        print(res)

        # Q7

        #На данных предыдущей задачи проверьте гипотезу об одинаковой средней температуре воздуха в дни,
        #когда уплотнительный кольца повреждались, и дни, когда повреждений не было.
        #Используйте перестановочный критерий и двустороннюю альтернативу. Чему равен достигаемый уровень значимости?
        #Округлите до четырёх знаков после десятичной точки.
        #
        #Чтобы получить такое же значение, как мы: установите random seed = 0; возьмите 10000 перестановок.

        np.random.seed(0)

        res = permutation_test(temp_bad, temp_good, max_permutations = 10000)
        print(res)

        return
    q67()

    return


def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples


def stat_intervals(stat, alpha=0.05):
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries


def proportions_confint_diff_ind_prob(p1, p2, n1, n2, alpha = 0.05):
    z = stats.norm.ppf(1 - alpha / 2.)

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1)/ n1 + p2 * (1 - p2)/ n2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1)/ n1 + p2 * (1 - p2)/ n2)

    return (left_boundary, right_boundary)


def proportions_diff_confint_ind(m1, n1, m2, n2, alpha = 0.05):
    z = stats.norm.ppf(1 - alpha / 2.)

    p1 = m1/n1
    p2 = m2/n2

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1)/ n1 + p2 * (1 - p2)/ n2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1)/ n1 + p2 * (1 - p2)/ n2)

    return (left_boundary, right_boundary)


def proportions_diff_confint_ind_prob(p1, p2, n1, n2, alpha = 0.05):
    z = stats.norm.ppf(1 - alpha / 2.)

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1)/ n1 + p2 * (1 - p2)/ n2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1)/ n1 + p2 * (1 - p2)/ n2)

    return (left_boundary, right_boundary)


def proportions_diff_z_stat_ind(m1, n1, m2, n2):
    p1 = m1/n1
    p2 = m2/n2
    P = float(p1*n1 + p2*n2) / (n1 + n2)

    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))


def proportions_diff_z_stat_ind_prob(p1, p2, n1, n2):
    P = float(p1*n1 + p2*n2) / (n1 + n2)
    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))


def proportions_diff_z_test(z_stat, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    if alternative == 'two-sided':
        return 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        return stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - stats.norm.cdf(z_stat)


def permutation_t_stat_ind(sample1, sample2):
    return np.mean(sample1) - np.mean(sample2)


def get_random_combinations(n1, n2, max_combinations):
    index = list(range(n1 + n2))
    indices = set([tuple(index)])
    for i in range(max_combinations - 1):
        np.random.shuffle(index)
        indices.add(tuple(index))
    return [(index[:n1], index[n1:]) for index in indices]


def permutation_zero_dist_ind(sample1, sample2, max_combinations = None):
    joined_sample = np.hstack((sample1, sample2))
    n1 = len(sample1)
    n = len(joined_sample)

    if max_combinations:
        indices = get_random_combinations(n1, len(sample2), max_combinations)
    else:
        indices = [(list(index), filter(lambda i: i not in index, range(n))) \
                    for index in itertools.combinations(range(n), n1)]

    distr = [joined_sample[list(i[0])].mean() - joined_sample[list(i[1])].mean() \
             for i in indices]
    return distr


def permutation_test(sample, mean, max_permutations = None, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    t_stat = permutation_t_stat_ind(sample, mean)

    zero_distr = permutation_zero_dist_ind(sample, mean, max_permutations)

    if alternative == 'two-sided':
        return sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'less':
        return sum([1. if x <= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        return sum([1. if x >= t_stat else 0. for x in zero_distr]) / len(zero_distr)