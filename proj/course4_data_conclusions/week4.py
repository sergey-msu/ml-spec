import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import scipy
import statsmodels.stats.multitest as smm
import statsmodels
from scipy import stats
from statsmodels.stats.weightstats import *


def header():
    return 'WEEK 4: Week of Problems';


def run():

    #bioinformatics()
    churn()

    return


def bioinformatics():

    print('\n------------ BIOINFORMATICS ------------\n')

    df = pd.read_csv(utils.PATH.COURSE_FILE(4, 'gene_high_throughput_sequencing.csv', 'week4'), index_col='Patient_id')

    print(df.head())
    print('\n')
    print(df.info())
    print(df.shape)

    n_genes = df.shape[1]-1

    df_normal = df[df['Diagnosis']=='normal'].values[:, 1:]
    df_early  = df[df['Diagnosis']=='early neoplasia'].values[:, 1:]
    df_cancer = df[df['Diagnosis']=='cancer'].values[:, 1:]

    p_values_normal, p_values_early = np.zeros((n_genes,)), np.zeros((n_genes,))


    def part1():
        for i in range(n_genes):
            normal_data = df_normal[:, i]
            early_data  = df_early[:, i]
            cancer_data = df_cancer[:, i]
            res_normal = stats.ttest_ind(normal_data, early_data, equal_var=False)
            res_early  = stats.ttest_ind(early_data, cancer_data, equal_var=False)
            p_values_normal[i] = res_normal[1]
            p_values_early[i]  = res_early[1]

        diffs_normal = p_values_normal<0.05
        diffs_early = p_values_early<0.05
        print(sum(diffs_normal), sum(diffs_early))

        utils.PATH.SAVE_RESULT((4, 4), (1, 1), int(sum(diffs_normal)))
        utils.PATH.SAVE_RESULT((4, 4), (1, 2), int(sum(diffs_early)))

        return
    part1()


    def part2():

        reject_normal, p_normal_corr, a1, a2 = smm.multipletests(p_values_normal, alpha=0.025, method='holm')
        reject_early, p_early_corr, a1, a2   = smm.multipletests(p_values_early, alpha=0.025, method='holm')

        confs_normals = 0
        for i, p in enumerate(p_normal_corr):
            if p>=0.025:
                continue

            control_data = df_normal[:, i]
            treatment_data = df_early[:, i]
            coeff = fold_change(control_data, treatment_data)
            if coeff>1.5:
                confs_normals += 1

        confs_earlies = 0
        for i, p in enumerate(p_early_corr):
            if p>=0.025:
                continue

            control_data = df_early[:, i]
            treatment_data = df_cancer[:, i]
            coeff = fold_change(control_data, treatment_data)
            if coeff>1.5:
                confs_earlies += 1

        print(confs_normals, confs_earlies)

        utils.PATH.SAVE_RESULT((4, 4), (2, 1), confs_normals)
        utils.PATH.SAVE_RESULT((4, 4), (2, 2), confs_earlies)

        return
    #part2()


    def part3():

        reject_normal, p_normal_corr, a1, a2 = smm.multipletests(p_values_normal, alpha=0.025, method='fdr_bh')
        reject_early, p_early_corr, a1, a2   = smm.multipletests(p_values_early, alpha=0.025, method='fdr_bh')

        confs_normals = 0
        for i, p in enumerate(p_normal_corr):
            if p>=0.025:
                continue

            control_data = df_normal[:, i]
            treatment_data = df_early[:, i]
            coeff = fold_change(control_data, treatment_data)
            if coeff>1.5:
                confs_normals += 1

        confs_earlies = 0
        for i, p in enumerate(p_early_corr):
            if p>=0.025:
                continue

            control_data = df_early[:, i]
            treatment_data = df_cancer[:, i]
            coeff = fold_change(control_data, treatment_data)
            if coeff>1.5:
                confs_earlies += 1

        print(confs_normals, confs_earlies)

        utils.PATH.SAVE_RESULT((4, 4), (3, 1), confs_normals)
        utils.PATH.SAVE_RESULT((4, 4), (3, 2), confs_earlies)

        return
    part3()

    return


def fold_change(control, treatment):
    c = control.mean()
    t = treatment.mean()
    return abs(t/c if t>=c else -c/t)


def churn():

    df = pd.read_csv(utils.PATH.COURSE_FILE(4, 'churn_analysis.csv', 'week4'), sep=';', index_col='id')
    print(df.info())
    print(df.shape)

    df_control = df[df['treatment']==1]
    print(df_control.shape)

    def q1234():

        p_vals = []
        p_vals_corr = []
        p_vals_fisher = []

        states = np.unique(df['state'].values)

        for i, s1 in enumerate(states):
            df_s1 = df_control[df_control['state']==s1]
            s1_churn = df_s1['churn'].value_counts().to_dict()
            s1_f, s1_t = s1_churn.get(False, 0), s1_churn.get(True, 0)

            for j, s2 in enumerate(states):
                if j>=i:
                    break

                df_s2 = df_control[df_control['state']==s2]
                s2_churn = df_s2['churn'].value_counts().to_dict()
                s2_f, s2_t = s2_churn.get(False, 0), s2_churn.get(True, 0)

                table = [[s2_t, s2_f], [s1_t, s1_f]]

                chi2 = stats.chi2_contingency(table, correction=False)
                p_val = chi2[1]
                p_vals.append(p_val)

                chi2_corr = stats.chi2_contingency(table, correction=True)
                p_val_corr = chi2_corr[1]
                p_vals_corr.append(p_val_corr)

                fisher = stats.fisher_exact(table)
                p_val_fisher = fisher[1]
                p_vals_fisher.append(p_val_fisher)

        p_vals = np.array(p_vals)
        p_vals_corr = np.array(p_vals_corr)
        p_vals_fisher = np.array(p_vals_fisher)

        print(sum(p_vals<0.05), sum(p_vals_corr<0.05), sum(p_vals_fisher<0.05))
        print(p_vals.mean(), p_vals_corr.mean(), p_vals_fisher.mean())

        return
    #q1234()

    def q56():

        plt.figure(figsize=(7,7))
        stats.probplot(df['day_calls'].values, dist="norm", plot=plt)
        plt.show()
        stats.probplot(df['mes_estim'].values, dist="norm", plot=plt)
        plt.show()

        pearson_corr = df[['day_calls', 'mes_estim']].corr(method='pearson')
        print(pearson_corr)
        pearson_corr = stats.pearsonr(df['day_calls'].values, df['mes_estim'].values)
        print(pearson_corr)

        spearman_corr = df[['day_calls', 'mes_estim']].corr(method='spearman')
        print(spearman_corr)
        spearman_corr = stats.spearmanr(df['day_calls'].values, df['mes_estim'].values)
        print(spearman_corr)

        return
    #q56()

    def q8():

        state_data = df_control['state'].values
        churn_data = df_control['churn'].values

        states = np.unique(df['state'].values)

        confusion_matrix = np.zeros((len(states), 2))

        for i, state in enumerate(states):
            df_state = df_control[df_control['state']==state]
            state_churn = df_state['churn'].value_counts().to_dict()
            confusion_matrix[i, 0] = state_churn.get(False, 0)
            confusion_matrix[i, 1] = state_churn.get(True, 0)

        # check matrix
        n_pi = confusion_matrix.sum(axis=1)
        n_pj = confusion_matrix.sum(axis=0)
        n = n_pj.sum()
        ni, nj = len(n_pi), len(n_pj)
        check_matrix = np.zeros((len(states), 2))

        for i in range(ni):
            for j in range(nj):
                check_matrix[i, j] = n_pi[i]*n_pj[j]/n

        print((check_matrix < 5).sum()/(ni*nj))


        cramers_coeff, p_val = cramers_corrected_stat(confusion_matrix)
        print(cramers_coeff, p_val)

        return
    q8()


    def q9():

        df_treat0 = df[df['treatment']==0]
        df_treat2 = df[df['treatment']==2]


        #ca_control = df_control[df_control['state']=='CA']['churn'].apply(lambda x: 1 if x else 0).values
        #ca_treat0  = df_treat0[df_treat0['state']=='CA']['churn'].apply(lambda x: 1 if x else 0).values
        #ca_treat2  = df_treat2[df_treat2['state']=='CA']['churn'].apply(lambda x: 1 if x else 0).values
        #
        ##ca_control_dist = ca_control.value_counts().to_dict() # 10 5
        ##ca_treat0_dist  = ca_treat0.value_counts().to_dict()  # 10 3
        ##ca_treat2_dist  = ca_treat2.value_counts().to_dict()  # 5  1
        #
        #table = [[5, 3], [10, 10]]
        #
        #p_val_control_to_0 = proportions_diff_z_test(proportions_diff_z_stat_ind(5, 15, 3, 13), alternative='two-sided')
        #p_val_control_to_0_chi2 = stats.chi2_contingency(table, correction=True)[1]
        #p_val_control_to_0_fisher = stats.fisher_exact(table)[1]

        #p_val_control_to_0_ttest = stats.ttest_ind(ca_control, ca_treat0, equal_var = False)[1]

        states = np.unique(df['state'].values)

        def do_stats(df1, df2):
            p_values = np.zeros_like(states)
            p_values_chi2 = np.zeros_like(states)
            p_values_fisher = np.zeros_like(states)
            diffs = np.zeros_like(states)

            for i, state in enumerate(states):

                # proportions

                df_state = df1[df1['state']==state]
                state_churn = df_state['churn'].value_counts().to_dict()
                state_t, state_f = state_churn.get(True, 0), state_churn.get(False, 0)
                m1 = state_t
                n1 = state_t + state_f

                df_treat_state = df2[df2['state']==state]
                treat_state_churn = df_treat_state['churn'].value_counts().to_dict()
                treat_state_t, treat_state_f = treat_state_churn.get(True, 0), treat_state_churn.get(False, 0)
                m2 = treat_state_t
                n2 = treat_state_t + treat_state_f

                p_values[i] = proportions_diff_z_test(proportions_diff_z_stat_ind(m1, n1, m2, n2), alternative='two-sided')
                diffs[i] = proportions_diff_confint_ind(m1, n1, m2, n2)

                table = [[state_t, treat_state_t],
                         [state_f, treat_state_f]]

                # chi-2 criteria

                chi2_corr = stats.chi2_contingency(table, correction=True)
                p_values_chi2[i] = chi2_corr[1]

                # fisher criteria

                fisher = stats.fisher_exact(table)
                p_values_fisher[i] = fisher[1]

            reject, p_corr, a1, a2 = smm.multipletests(p_values, alpha=0.05, method='holm')
            print('Proportions:', sum(p_corr<0.05))

            reject, p_corr, a1, a2 = smm.multipletests(p_values_chi2, alpha=0.05, method='fdr_bh')
            print('Chi^2:', sum(p_corr<0.05))

            reject, p_corr, a1, a2 = smm.multipletests(p_values_fisher, alpha=0.05, method='fdr_bh')
            print('Fisher:', sum(p_corr<0.05))

            return

        do_stats(df_control, df_treat0)
        do_stats(df_control, df_treat0)
        do_stats(df_treat0,  df_treat2)

        ### DID NOT PASSED ###

        return
    q9()

    return


def cramers_corrected_stat(confusion_matrix):
    res = stats.chi2_contingency(confusion_matrix)
    chi2 = res[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / (min(k, r) - 1)), res[1]


def proportions_diff_confint_ind(m1, n1, m2, n2, alpha = 0.05):
    z = scipy.stats.norm.ppf(1 - alpha / 2.)

    p1 = m1 / n1
    p2 = m2 / n2

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1)/n1 + p2 * (1 - p2)/n2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1)/n1 + p2 * (1 - p2)/n2)

    return (left_boundary, right_boundary)


def proportions_diff_z_stat_ind(m1, n1, m2, n2):
    p1 = m1 / n1
    p2 = m2 / n2
    P = float(p1*n1 + p2*n2) / (n1 + n2)

    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))


def proportions_diff_z_test(z_stat, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    if alternative == 'two-sided':
        return 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        return scipy.stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - scipy.stats.norm.cdf(z_stat)