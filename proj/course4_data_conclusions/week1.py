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
    return 'WEEK 1: Intervals and Hypotheses';


def run():

    #quiz1()
    #quiz2()
    #quiz3()
    quiz4()

    #test()

    return


def quiz1():

    df = pd.read_csv(utils.PATH.COURSE_FILE(4, 'water.txt', 'week1'), sep='\t')
    print(df.head())

    #df.plot(x='hardness', y='mortality', kind='scatter')
    #plt.show()

    # Q2

    n = len(df)
    mortality = df['mortality'].values
    mean = mortality.mean()
    sigma = mortality.std(ddof=1)/math.sqrt(n)
    print('all', _tconfint_generic(mean, sigma, n-1, 0.05, 'two-sided'))


    # Q3

    south_df = df[df['location']=='South']
    n_south = len(south_df)
    mortality = south_df['mortality'].values
    mean = mortality.mean()
    sigma = mortality.std(ddof=1)/math.sqrt(n_south)
    print('mortality south', _tconfint_generic(mean, sigma, n_south-1, 0.05, 'two-sided'))

    # Q4

    north_df = df[df['location']=='North']
    n_north = len(north_df)
    mortality = north_df['mortality'].values
    mean = mortality.mean()
    sigma = mortality.std(ddof=1)/math.sqrt(n_north)
    print('mortality north', _tconfint_generic(mean, sigma, n_north-1, 0.05, 'two-sided'))

    # Q5

    hardness_south = south_df['hardness'].values
    mean_south = hardness_south.mean()
    sigma_south = hardness_south.std(ddof=1)/math.sqrt(n_south)

    hardness_north = north_df['hardness'].values
    mean_north = hardness_north.mean()
    sigma_north = hardness_north.std(ddof=1)/math.sqrt(n_north)


    print('hardeness north', _tconfint_generic(mean_north, sigma_north, n_north-1, 0.05, 'two-sided'))
    print('hardeness south', _tconfint_generic(mean_south, sigma_south, n_south-1, 0.05, 'two-sided'))

    # Q6

    print((10*norm.ppf(1-0.05/2))**2)


    return


def quiz2():

    normal_interval = proportion_confint(1, 50, alpha=0.05, method = 'normal')
    print(normal_interval)

    wilson_interval = proportion_confint(1, 50, alpha=0.05, method = 'wilson')
    print(wilson_interval)

    n_samples = samplesize_confint_proportion(0.02, half_length=0.01, alpha=0.05)
    n_samples = int(np.ceil(n_samples)) # интервал ширины 0.02
    print(n_samples)

    ps = np.linspace(0.01, 0.99, 100)
    ns = []
    for p in ps:
        n = samplesize_confint_proportion(p, half_length=0.01, alpha=0.05)
        n = int(np.ceil(n))
        ns.append(n)

    plt.plot(ps, ns)
    plt.show()


    n_samples = samplesize_confint_proportion(0.5, half_length=0.01, alpha=0.05)
    n_samples = int(np.ceil(n_samples)) # интервал ширины 0.02
    print(n_samples)

    return


def quiz3():

    print(norm.ppf(1-0.003/2))

    p1 = 104/11037.0
    p2 = 189/11034.0

    print(p2-p1)

    def proportions_confint_diff_ind(m1, n1, m2, n2, alpha = 0.05):
        z = norm.ppf(1 - alpha / 2.)

        p1 = m1/float(n1)
        p2 = m2/float(n2)
        left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1)/ n1 + p2 * (1 - p2)/ n2)
        right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1)/ n1 + p2 * (1 - p2)/ n2)

        return (left_boundary, right_boundary)

    left_boundary, right_boundary = proportions_confint_diff_ind(189, 11034, 104, 11037)
    print(left_boundary, right_boundary)

    chance1 = p1/(1-p1)
    chance2 = p2/(1-p2)
    print(chance1, chance2, chance2/chance1)

    # Q8

    print('--------------- Q8 ---------------')

    data_asp = np.array([1]*104 + [0]*(11037-104))
    data_pla = np.array([1]*189 + [0]*(11034-189))

    data_asp_bootstrap = get_bootstrap_samples(data_asp, 1000)
    p_asp = data_asp_bootstrap.mean(axis=1)
    chances_asp = p_asp/(1-p_asp)

    data_pla_bootstrap = get_bootstrap_samples(data_pla, 1000)
    p_pla = data_pla_bootstrap.mean(axis=1)
    chances_pla = p_pla/(1-p_pla)

    chances = chances_pla/chances_asp

    answer = stat_intervals(chances)

    print(answer)

    return


def quiz4():

    # Q1,2

    def q12():
        p = 0.75
        n = 100
        m = 67

        F_H0 = stats.binom(n, p)

        x = np.linspace(0, n, n+1)
        plt.bar(x, F_H0.pmf(x), align='center')
        x2 = np.linspace(m, n, n-m+1)
        plt.bar(x2, F_H0.pmf(x2), align='center', color='red')

        pvalue = stats.binom_test(m, n, p, alternative='two-sided')
        print(pvalue)

        plt.show()
        return
    #q12()

    # Q3

    df = pd.read_csv(utils.PATH.COURSE_FILE(4, 'pines.txt', 'week1'), sep='\t')
    print(df.head())

    x = df['sn'].values
    y = df['we'].values
    st = binned_statistic_2d(x, y, None, bins=5, statistic='count')
    print(st.statistic)

    expected_mean = st.statistic.sum()/25
    print(expected_mean)

    # Q4

    observed_freqs = st.statistic.reshape((25))
    expected_freqs = expected_mean*np.ones((25))
    cs = stats.chisquare(observed_freqs, expected_freqs, ddof = 1)

    observed_freqs = np.array([1, 2, 1, 1, 2, 3, 1, 1, 1])
    expected_freqs = observed_freqs.mean()*np.ones_like(observed_freqs)
    cs = stats.chisquare(observed_freqs, expected_freqs, ddof = 1)

    print(cs)

    return


def test():

    x = np.random.randint(0, 2, size=(100000, 1000))

    v = np.abs(np.diff(x, axis=1)).sum(axis=1)

    plt.hist(v, bins=50)
    plt.show()

    return


def get_bootstrap_samples(data, n_samples):
    np.random.seed(0)
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