import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score


def header():
    return 'WEEK 2: Metrics and SKLearn Intro';


def run():

    homework()

    return


def homework():

    actual_0 = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                          1.,  1.,  1., 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
    predicted_0 = np.array([ 0.19015288,  0.23872404,  0.42707312,  0.15308362,  0.2951875,
                0.23475641,  0.17882447,  0.36320878,  0.33505476,  0.202608  ,
                0.82044786,  0.69750253,  0.60272784,  0.9032949 ,  0.86949819,
                0.97368264,  0.97289232,  0.75356512,  0.65189193,  0.95237033,
                0.91529693,  0.8458463 ])

    #plt.figure(figsize=(5, 5))
    #scatter(actual_0, predicted_0, 0.5)
    #plt.show()

    actual_1 = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                    0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                    1.,  1.,  1.,  1.])
    predicted_1 = np.array([ 0.41310733,  0.43739138,  0.22346525,  0.46746017,  0.58251177,
                0.38989541,  0.43634826,  0.32329726,  0.01114812,  0.41623557,
                0.54875741,  0.48526472,  0.21747683,  0.05069586,  0.16438548,
                0.68721238,  0.72062154,  0.90268312,  0.46486043,  0.99656541,
                0.59919345,  0.53818659,  0.8037637 ,  0.272277  ,  0.87428626,
                0.79721372,  0.62506539,  0.63010277,  0.35276217,  0.56775664])
    actual_2 = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,
                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
    predicted_2 = np.array([ 0.07058193,  0.57877375,  0.42453249,  0.56562439,  0.13372737,
                0.18696826,  0.09037209,  0.12609756,  0.14047683,  0.06210359,
                0.36812596,  0.22277266,  0.79974381,  0.94843878,  0.4742684 ,
                0.80825366,  0.83569563,  0.45621915,  0.79364286,  0.82181152,
                0.44531285,  0.65245348,  0.69884206,  0.69455127])

    # рискующий идеальный алгоитм
    actual_0r = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,
                1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
    predicted_0r = np.array([ 0.23563765,  0.16685597,  0.13718058,  0.35905335,  0.18498365,
                0.20730027,  0.14833803,  0.18841647,  0.01205882,  0.0101424 ,
                0.10170538,  0.94552901,  0.72007506,  0.75186747,  0.85893269,
                0.90517219,  0.97667347,  0.86346504,  0.72267683,  0.9130444 ,
                0.8319242 ,  0.9578879 ,  0.89448939,  0.76379055])
    # рискующий хороший алгоритм
    actual_1r = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
                1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
    predicted_1r = np.array([ 0.13832748,  0.0814398 ,  0.16136633,  0.11766141,  0.31784942,
                0.14886991,  0.22664977,  0.07735617,  0.07071879,  0.92146468,
                0.87579938,  0.97561838,  0.75638872,  0.89900957,  0.93760969,
                0.92708013,  0.82003675,  0.85833438,  0.67371118,  0.82115125,
                0.87560984,  0.77832734,  0.7593189,  0.81615662,  0.11906964,
                0.18857729])

    actual_10 = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                1.,  1.,  1.])
    predicted_10 = np.array([ 0.29340574, 0.47340035,  0.1580356 ,  0.29996772,  0.24115457,  0.16177793,
                             0.35552878,  0.18867804,  0.38141962,  0.20367392,  0.26418924, 0.16289102,
                             0.27774892,  0.32013135,  0.13453541, 0.39478755,  0.96625033,  0.47683139,
                             0.51221325,  0.48938235, 0.57092593,  0.21856972,  0.62773859,  0.90454639,  0.19406537,
                             0.32063043,  0.4545493 ,  0.57574841,  0.55847795 ])
    actual_11 = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                    0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
    predicted_11 = np.array([ 0.35929566, 0.61562123,  0.71974688,  0.24893298,  0.19056711,  0.89308488,
                0.71155538,  0.00903258,  0.51950535,  0.72153302,  0.45936068,  0.20197229,  0.67092724,
                             0.81111343,  0.65359427,  0.70044585,  0.61983513,  0.84716577,  0.8512387 ,
                             0.86023125,  0.7659328 ,  0.70362246,  0.70127618,  0.8578749 ,  0.83641841,
                             0.62959491,  0.90445368])

    T = 0.65
    precision_1 = precision_score(actual_1, predicted_1>T)
    recall_1 = recall_score(actual_1, predicted_1>T)
    precision_10 = precision_score(actual_10, predicted_10>T)
    recall_10 = recall_score(actual_10, predicted_10>T)
    precision_11 = precision_score(actual_11, predicted_11>T)
    recall_11 = recall_score(actual_11, predicted_11>T)
    print(precision_1, recall_1, precision_10, recall_10, precision_11, recall_11)

    ks = np.linspace(0, 10, num=11, dtype=int)
    k_max  = [0, 0, 0]
    f1_max = [-1, -1, -1]
    for k in ks:
        T = 0.1*k
        f1_1 = f1_score(actual_1, predicted_1>T)
        if (f1_1 > f1_max[0]):
            f1_max[0] = f1_1
            k_max[0] = k
        f1_10 = f1_score(actual_10, predicted_10>T)
        if (f1_10 > f1_max[1]):
            f1_max[1] = f1_10
            k_max[1] = k
        f1_11 = f1_score(actual_11, predicted_11>T)
        if (f1_11 > f1_max[2]):
            f1_max[2] = f1_11
            k_max[2] = k

    print(k_max)

    wll_0  = weighted_log_loss(actual_0, predicted_0)
    wll_1  = weighted_log_loss(actual_1, predicted_1)
    wll_2  = weighted_log_loss(actual_2, predicted_2)
    wll_0r = weighted_log_loss(actual_0r, predicted_0r)
    wll_1r = weighted_log_loss(actual_1r, predicted_1r)
    wll_10 = weighted_log_loss(actual_10, predicted_10)
    wll_11 = weighted_log_loss(actual_11, predicted_11)

    print(wll_0, wll_1, wll_2, wll_0r, wll_1r, wll_10, wll_11)


    fpr_0, tpr_0, thr_0 = roc_curve(actual_0, predicted_0)
    fpr_1, tpr_1, thr_1 = roc_curve(actual_1, predicted_1)
    fpr_2, tpr_2, thr_2 = roc_curve(actual_2, predicted_2)
    fpr_0r, tpr_0r, thr_0r = roc_curve(actual_0r, predicted_0r)
    fpr_1r, tpr_1r, thr_1r = roc_curve(actual_1r, predicted_1r)
    fpr_10, tpr_10, thr_10 = roc_curve(actual_10, predicted_10)
    fpr_11, tpr_11, thr_11 = roc_curve(actual_11, predicted_11)
    T_0 = get_optimal_threshold(fpr_0, tpr_0, thr_0)
    T_1 = get_optimal_threshold(fpr_1, tpr_1, thr_1)
    T_2 = get_optimal_threshold(fpr_2, tpr_2, thr_2)
    T_0r = get_optimal_threshold(fpr_0r, tpr_0r, thr_0r)
    T_1r = get_optimal_threshold(fpr_1r, tpr_1r, thr_1r)
    T_10 = get_optimal_threshold(fpr_10, tpr_10, thr_10)
    T_11 = get_optimal_threshold(fpr_11, tpr_11, thr_11)

    print(T_0, T_1, T_2, T_0r, T_1r, T_10, T_11)

    return

def weighted_log_loss(actual, predicted, p=0.3):
    return -(p*actual*np.log(predicted) + (1 - p)*(1 - actual)*np.log(1 - predicted)).mean()


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
