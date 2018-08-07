import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
from sklearn.cluster import MeanShift


def header():
    return 'WEEK 1: Intro and Clusterization';


def run():

    file_name = utils.PATH.COURSE_FILE(3, 'week1//checkins.csv')
    if not os.path.exists(file_name):
        with open(utils.PATH.COURSE_FILE(3, 'week1//checkins.dat'), 'r') as src_file, \
             open(file_name, 'w') as trg_file:
            for line in src_file:
                if ('----' in line): continue
                if ('rows' in line): continue
                line = line.strip().replace('|', ',').replace(' ', '')
                trg_file.write(line+'\n')

    df = pd.read_csv(file_name)
    df.dropna(inplace=True)
    print(df.head())
    print(df.shape)
    #df['created_at'] = df['created_at'].apply(lambda x: x[:10]+' '+x[10:]).apply(pd.to_datetime)

    X = df[['latitude', 'longitude']].values[:100000, :]

    clust = MeanShift(bandwidth=0.1, n_jobs=-1)
    clust.fit(X)

    offices = np.array([[33.751277, -118.188740],
                        [25.867736, -80.324116],
                        [51.503016, -0.075479],
                        [52.378894, 4.885084],
                        [39.366487, 117.036146],
                        [-33.868457, 151.205134]])
    min_centers = []
    for label in np.unique(clust.labels_):
        X_label = X[clust.labels_==label, :]
        if X_label.shape[0] <= 15:
            continue

        center = clust.cluster_centers_[label]
        min_dist = np.Infinity
        min_center = None
        for office in offices:
            d = np.linalg.norm(center - office)
            if d<min_dist:
                min_dist = d
                min_center = center
        min_centers.append([min_center, min_dist])

    min_centers = np.array(min_centers)
    idxs = np.argpartition(min_centers[:, 1], 20)[:20]
    best_centers = min_centers[idxs, 0]

    print(best_centers[0])

    return
