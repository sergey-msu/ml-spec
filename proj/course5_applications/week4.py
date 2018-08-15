import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from collections import Counter


def header():
    return 'WEEK 4: Ranging';


def run():

    train_data = data_from_file('coursera_sessions_train.txt')
    test_data = data_from_file('coursera_sessions_test.txt')

    populars, bestsellers = get_freqs(train_data)

    print(populars.most_common(5))
    print(bestsellers.most_common(5))

    (r1, p1), (r5, p5) = average_metrics(1, train_data, populars), average_metrics(5, train_data, populars)
    print(r1, p1, r5, p5)
    utils.PATH.SAVE_RESULT((5, 4), (1, 1), [r1, p1, r5, p5])

    (r1, p1), (r5, p5) = average_metrics(1, test_data, populars), average_metrics(5, test_data, populars)
    print(r1, p1, r5, p5)
    utils.PATH.SAVE_RESULT((5, 4), (1, 2), [r1, p1, r5, p5])

    (r1, p1), (r5, p5) = average_metrics(1, train_data, bestsellers), average_metrics(5, train_data, bestsellers)
    print(r1, p1, r5, p5)
    utils.PATH.SAVE_RESULT((5, 4), (1, 3), [r1, p1, r5, p5])

    (r1, p1), (r5, p5) = average_metrics(1, test_data, bestsellers), average_metrics(5, test_data, bestsellers)
    print(r1, p1, r5, p5)
    utils.PATH.SAVE_RESULT((5, 4), (1, 4), [r1, p1, r5, p5])

    return


def data_from_file(fname):
    result = []

    with open(utils.PATH.COURSE_FILE(5, fname, 'week4')) as file:
        for line in file:
            segs = line.strip().split(';')
            seen = np.array(segs[0].split(',')).astype(np.int)
            bght = None if len(segs[1])==0 else np.array(segs[1].split(',')).astype(np.int)
            result.append((seen, bght))

    return result


def get_freqs(data):
    seen_freqs = Counter()
    bgth_freqs = Counter()

    for line in data:
        seen_freqs.update(line[0])
        if line[1] is not None:
            bgth_freqs.update(line[1])

    return seen_freqs, bgth_freqs


def average_metrics(k, data, freqs):
    precision = []
    recall = []
    vfunc = np.vectorize(lambda x: freqs[x])

    for line in data:
        bgth = line[1]
        if bgth is None:
            continue

        seen = line[0]
        _, idx = np.unique(seen, return_index=True)
        seen_dist = seen[np.sort(idx)]
        n_rec = min(k, len(seen_dist))

        recom = sort_recoms(seen_dist, freqs, n_rec)

        bought = len(set(recom).intersection(set(bgth)))
        precision.append(bought/k)
        recall.append(bought/len(bgth))

    return round(np.array(recall).mean(), 2), \
           round(np.array(precision).mean(), 2)


def sort_recoms(seen, freqs, n_rec):
    seen = list(seen)
    n_rec = min(n_rec, len(seen))
    recoms = []

    for i in range(n_rec):
        j_max = -1
        v_max = -1
        for j, s in enumerate(seen):
            v = freqs[s]
            if v>v_max:
                v_max = v
                j_max = j
        recoms.append(seen.pop(j_max))

    return recoms