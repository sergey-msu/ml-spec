import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import utils
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.model_selection import cross_val_score
from sklearn import datasets


def header():
    return 'WEEK 2: PCA';


def run():

    #pca_example()
    homework()

    return


def homework():

    def Q1():
        df = pd.read_csv(utils.PATH.COURSE_FILE(3, 'week2//data_task1.csv'))
        print(df.head())

        D = df.shape[1]
        d_opt = D
        score_opt = -np.Infinity
        scores = []

        for d in range(1, D+1):
            model = PCA(n_components=d)
            score = cross_val_score(model, df).mean()
            scores.append(score)
            if score > score_opt:
                score_opt = score
                d_opt = d
            print(d, score)

        print(d_opt)
        utils.PATH.SAVE_RESULT((3, 2), (1, 1), d_opt)

        plot_scores(np.array(scores))
        return
    #Q1()


    def Q2():
        df = pd.read_csv(utils.PATH.COURSE_FILE(3, 'week2//data_task2.csv'))
        #print(df.head())

        model = PCA(n_components=df.shape[1])
        X_new = model.fit_transform(df)

        vars = np.var(X_new, axis=0)
        var_diffs = -np.diff(vars)
        d_opt = np.argmax(var_diffs) + 1
        print(d_opt)
        utils.PATH.SAVE_RESULT((3, 2), (1, 2), d_opt)

        plot_variances(vars)
        return
    #Q2()


    def Q3():

        data = datasets.load_iris()
        X = data.data
        y = data.target

        model = PCA(n_components=4)
        Z = model.fit_transform(X)

        A = np.concatenate((X, Z[:, :2]), axis=1)
        print(A.shape)

        R = np.round_(np.corrcoef(A, rowvar=False), 2)
        print(R)

        utils.PATH.SAVE_RESULT((3, 2), (1, 3), '1 3 4 2')

        return
    #Q3()


    def Q4():

        data = datasets.fetch_olivetti_faces(shuffle=True, random_state=0)
        X = data.data
        y = data.target

        image_shape = (64, 64)

        n = X.shape[0]
        n_components = 10

        model = RandomizedPCA(n_components=n_components)
        Z = model.fit_transform(X)
        Z_c = Z #- Z.mean(axis=1).reshape((n, 1))  !!!! ERROR IN COURSERA
        Z_c = Z_c*Z_c
        Z_tot = Z_c.sum(axis=1).reshape((n, 1))

        Cos = Z_c / Z_tot

        i_s = []
        for j in range(n_components):
            i = np.argmax(Cos[:, j])
            i_s.append(i)
            image = X[i, :].reshape(image_shape)
            #plt.imshow(image)
            #plt.show()

        utils.PATH.SAVE_RESULT((3, 2), (1, 4), i_s)

        return
    Q4()


    return


def pca_example():
    mu = np.zeros(2)
    C = np.array([[3, 1], [1, 2]])

    data = np.random.multivariate_normal(mu, C, size=50)
    plt.scatter(data[:, 0], data[:, 1])
    #plt.show()

    v, W_true = np.linalg.eig(C)
    plt.plot(data[:, 0], (W_true[0, 0]/W_true[0, 1])*data[:, 0], color='g')
    plt.plot(data[:, 0], (W_true[1, 0]/W_true[1, 1])*data[:, 0], color='g')
    g_patch = mpatches.Patch(color='g', label='True components')
    plt.legend(handles=[g_patch])
    plt.axis('equal')
    limits = [np.minimum(np.amin(data[:,0]), np.amin(data[:,1])),
              np.maximum(np.amax(data[:,0]), np.amax(data[:,1]))]
    plt.xlim(limits[0],limits[1])
    plt.ylim(limits[0],limits[1])
    #plt.show()

    model = PCA(n_components=2)
    model.fit(data)

    plt.scatter(data[:,0], data[:,1])
    # построим истинные компоненты, вдоль которых максимальна дисперсия данных
    plt.plot(data[:,0], (W_true[0,0]/W_true[0,1])*data[:,0], color="g")
    plt.plot(data[:,0], (W_true[1,0]/W_true[1,1])*data[:,0], color="g")
    # построим компоненты, полученные с использованием метода PCA:
    plot_principal_components(data, model, scatter=False, legend=False)
    c_patch = mpatches.Patch(color='c', label='Principal components')
    plt.legend(handles=[g_patch, c_patch])
    #plt.show()

    data_large = np.random.multivariate_normal(mu, C, size=5000)

    model = PCA(n_components=2)
    model.fit(data_large)
    plt.scatter(data_large[:,0], data_large[:,1], alpha=0.1)
    # построим истинные компоненты, вдоль которых максимальна дисперсия данных
    plt.plot(data_large[:,0], (W_true[0,0]/W_true[0,1])*data_large[:,0], color="g")
    plt.plot(data_large[:,0], (W_true[1,0]/W_true[1,1])*data_large[:,0], color="g")
    # построим компоненты, полученные с использованием метода PCA:
    plot_principal_components(data_large, model, scatter=False, legend=False)
    c_patch = mpatches.Patch(color='c', label='Principal components')
    plt.legend(handles=[g_patch, c_patch])

    data_large = np.random.multivariate_normal(mu, C, size=5000)

    model = PCA(n_components=2)
    model.fit(data_large)
    plt.scatter(data_large[:,0], data_large[:,1], alpha=0.1)
    # построим истинные компоненты, вдоль которых максимальна дисперсия данных
    plt.plot(data_large[:,0], (W_true[0,0]/W_true[0,1])*data_large[:,0], color="g")
    plt.plot(data_large[:,0], (W_true[1,0]/W_true[1,1])*data_large[:,0], color="g")
    # построим компоненты, полученные с использованием метода PCA:
    plot_principal_components(data_large, model, scatter=False, legend=False)
    c_patch = mpatches.Patch(color='c', label='Principal components')
    plt.legend(handles=[g_patch, c_patch])
    plt.show()

    return


def plot_principal_components(data, model, scatter=True, legend=True):
    W_pca = model.components_
    if scatter:
        plt.scatter(data[:,0], data[:,1])
    plt.plot(data[:,0], -(W_pca[0,0]/W_pca[0,1])*data[:,0], color="c")
    plt.plot(data[:,0], -(W_pca[1,0]/W_pca[1,1])*data[:,0], color="c")
    if legend:
        c_patch = mpatches.Patch(color='c', label='Principal components')
        plt.legend(handles=[c_patch], loc='lower right')
    # сделаем графики красивыми:
    plt.axis('equal')
    limits = [np.minimum(np.amin(data[:,0]), np.amin(data[:,1]))-0.5,
              np.maximum(np.amax(data[:,0]), np.amax(data[:,1]))+0.5]
    plt.xlim(limits[0],limits[1])
    plt.ylim(limits[0],limits[1])
    plt.draw()


def plot_scores(d_scores):
    n_components = np.arange(1,d_scores.size+1)
    plt.plot(n_components, d_scores, 'b', label='PCA scores')
    plt.xlim(n_components[0], n_components[-1])
    plt.xlabel('n components')
    plt.ylabel('cv scores')
    plt.legend(loc='lower right')
    plt.show()


def plot_variances(d_variances):
    n_components = np.arange(1,d_variances.size+1)
    plt.plot(n_components, d_variances, 'b', label='Component variances')
    plt.xlim(n_components[0], n_components[-1])
    plt.xlabel('n components')
    plt.ylabel('variance')
    plt.legend(loc='upper right')
    plt.show()


def plot_iris(transformed_data, target, target_names):
    plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
        plt.scatter(transformed_data[target == i, 0],
                    transformed_data[target == i, 1], c=c, label=target_name)
    plt.legend()
    plt.show()