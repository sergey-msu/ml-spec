import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import statsmodels
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.diagnostic import het_breuschpagan


def header():
    return 'WEEK 3: Finding patterns in data';


def run():

    #quiz1()
    #quiz2()
    #quiz3()
    quiz4()

    return


def quiz1():

    def q45():

        df = pd.read_csv(utils.PATH.COURSE_FILE(4, 'illiteracy.txt', 'week3'), sep='\t')
        print(df.head())

        pearson_corr = df[['Illit', 'Births']].corr(method='pearson')
        spearman_corr = df[['Illit', 'Births']].corr(method='spearman')
        print(pearson_corr, spearman_corr)

        return
    q45()

    return


def quiz2():

    def q123():

        df = pd.read_csv(utils.PATH.COURSE_FILE(4, 'water.txt', 'week3'), sep='\t')
        print(df.head())

        peason_corr = df[['mortality', 'hardness']].corr(method='pearson')
        spearman_corr = df[['mortality', 'hardness']].corr(method='spearman')

        print(round(peason_corr, 4))
        print(round(spearman_corr, 4))

        df_south = df[df['location']=='South']
        df_north = df[df['location']=='North']

        pearson_corr_south = df_south[['mortality', 'hardness']].corr(method='pearson').values
        pearson_corr_north = df_north[['mortality', 'hardness']].corr(method='pearson').values

        print(round(pearson_corr_south[0, 1], 4), round(pearson_corr_north[0, 1], 4))

        return
    #q123()

    def q4567():

        # Q4

        w1 = 203
        m1 = 239
        w2 = 718
        m2 = 515

        matthews_coeff = (m2*w1 - m1*w2)/math.sqrt((m2 + w2)*(m2 + m1)*(w2 + w1)*(m1 + w1))
        print(round(matthews_coeff, 4))

        # Q5
        res = stats.chi2_contingency([[m2, w2], [m1, w1]])
        print(res)

        # Q6
        nw = w1 + w2
        nm = m1 + m2
        p1 = w1/nw # woman drinkers
        p2 = m1/nm # man drinkers

        res = proportions_diff_confint_ind_prob(p1, p2, nw, nm)
        print(res)

        # Q7
        res = proportions_diff_z_test(proportions_diff_z_stat_ind(p1, p2, nw, nm))
        print(res)

        return
    q4567()

    def q8910():
        inc = np.array([[197, 111, 33],
                       [382, 685, 331],
                       [110, 342, 333]])
        res = stats.chi2_contingency(inc)
        print(res[0], res[1])

        print('Cramer:', math.sqrt(res[0]/(inc.sum()*2)))

        return
    #q8910()

    return


def quiz3():

    df = pd.read_csv(utils.PATH.COURSE_FILE(4, 'AUCs.txt', 'week3'), sep='\t')
    print(df.head())

    # Q1234

    model_names = ['C4.5', 'C4.5+m', 'C4.5+cf', 'C4.5+m+cf']
    wilc_stats = []

    for i, first_model_name in enumerate(model_names):
        for j, second_model_name in enumerate(model_names):
            if j>=i:
                continue

            first_model_data  = df[first_model_name].values
            second_model_data = df[second_model_name].values

            wilc_stat = stats.wilcoxon(first_model_data, second_model_data)
            wilc_stats.append([first_model_name, second_model_name, round(wilc_stat[0], 4), round(wilc_stat[1], 4)])

    wilc_stats = pd.DataFrame.from_records(wilc_stats)
    wilc_stats.columns = ['Model A', 'Model B', 'Stat', 'p_value']
    print(wilc_stats.head())

    # Q5

    reject, p_corrected, a1, a2 = multipletests(wilc_stats['p_value'],
                                                alpha = 0.05,
                                                method = 'holm')

    wilc_stats['p_corrected'] = p_corrected
    wilc_stats['reject'] = reject

    print(wilc_stats.head(20))
    print(wilc_stats['reject'].value_counts())

    # Q6

    reject, p_corrected, a1, a2 = multipletests(wilc_stats['p_value'],
                                                alpha = 0.05,
                                                method = 'fdr_bh')

    wilc_stats['p_corrected'] = p_corrected
    wilc_stats['reject'] = reject

    print(wilc_stats.head(20))
    print(wilc_stats['reject'].value_counts())


    return


def quiz4():

    df = pd.read_csv(utils.PATH.COURSE_FILE(4, 'botswana.tsv', 'week3'), sep='\t')
    print(df.head())
    print(df.info())

    # Q1

    religion = df['religion'].value_counts()
    print('---- Q1:', religion)

    # Q2

    df_nonna = df.dropna()
    print('---- Q2:', df_nonna.shape)

    # Q3

    df['nevermarr'] = df['agefm'].apply(lambda x: 0 if x>=0 else 1)
    df.drop(['evermarr'], axis=1, inplace=True)
    df['agefm'].fillna(0, inplace=True)
    df['heduc'] = df.apply(lambda row: row['heduc'] if row['nevermarr']==0 else -1, axis=1)

    heduc_na = df['heduc'].isnull().value_counts()
    print('---- Q3:', heduc_na)

    # Q4

    df['idlnchld_noans'] = df['idlnchld'].apply(lambda x: 0 if x>=0 else -1)
    df['idlnchld'].fillna(-1, inplace=True)
    df['heduc_noans'] = df['heduc'].apply(lambda x: 0 if x>=0 else -1)
    df['heduc'].fillna(-1, inplace=True)
    df['usemeth_noans'] = df['usemeth'].apply(lambda x: 0 if x>=0 else -1)
    df['usemeth'].fillna(-1, inplace=True)

    df.dropna(inplace=True)
    print(df.shape)
    print('---- Q4:', df.shape[0]*df.shape[1])

    # Q5

    model = smf.ols('ceb ~ age + educ + religion + idlnchld + knowmeth + usemeth + agefm + heduc + urban + electric + radio + tv + bicycle +'\
                          'nevermarr + idlnchld_noans + heduc_noans + usemeth_noans', data=df)
    fitted = model.fit()
    sum = fitted.summary()
    print('---- Q5"\n', sum)

    # Q6

    print('---- Q6: p=%f' % het_breuschpagan(fitted.resid, fitted.model.exog)[1])

    # Q7


    model1 = smf.ols('ceb ~ age + educ + religion + idlnchld + knowmeth + usemeth + agefm + heduc + urban + electric + radio + tv + bicycle +'\
                           'nevermarr + idlnchld_noans + heduc_noans + usemeth_noans', data=df)
    fitted = model1.fit(cov_type='HC1')
    sum = fitted.summary()
    print('---- Q7\n:', sum)

    # Q8

    model2 = smf.ols('ceb ~ age + educ + idlnchld + knowmeth + usemeth + agefm + heduc + urban + electric + bicycle +'\
                           'nevermarr + idlnchld_noans + heduc_noans + usemeth_noans', data=df)
    fitted = model2.fit()

    print('p=%f' % het_breuschpagan(fitted.resid, fitted.model.exog)[1])

    model2 = smf.ols('ceb ~ age + educ + idlnchld + knowmeth + usemeth + agefm + heduc + urban + electric + bicycle +'\
                           'nevermarr + idlnchld_noans + heduc_noans + usemeth_noans', data=df)
    fitted = model2.fit(cov_type='HC1')

    print("---- Q8: F=%f, p=%f, k1=%f" % model1.fit().compare_f_test(model2.fit()))

    # Q9

    model3 = smf.ols('ceb ~ age + educ + idlnchld + knowmeth + agefm + heduc + urban + electric + bicycle +'\
                           'nevermarr + idlnchld_noans + heduc_noans', data=df)
    fitted = model3.fit()
    print("---- Q9: F=%f, p=%f, k1=%f" % model2.fit().compare_f_test(model3.fit()))

    # Q10

    model = smf.ols('ceb ~ age + educ + idlnchld + knowmeth + agefm + heduc + urban + electric + bicycle +'\
                           'nevermarr + idlnchld_noans + heduc_noans', data=df)
    fitted = model.fit(cov_type='HC1')
    sum = fitted.summary()
    print('---- Q10\n:', sum)


    return


def proportions_diff_confint_ind_prob(p1, p2, n1, n2, alpha = 0.05):
    z = stats.norm.ppf(1 - alpha / 2.)

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1)/ n1 + p2 * (1 - p2)/ n2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1)/ n1 + p2 * (1 - p2)/ n2)

    return (left_boundary, right_boundary)


def proportions_diff_z_stat_ind(p1, p2, n1, n2):
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