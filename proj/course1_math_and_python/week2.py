import re
import codecs
import numpy as np
import utils
from matplotlib import pyplot as plt
from collections import Counter
from scipy.spatial.distance import cosine


def header():
    return 'WEEK 2: Python and Linear Algebra';


def run():

    #numpy_test()
    #matplotlib_test()
    #homework1()
    homework2()

    return


def numpy_test():
    x = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    print(x[1:, :2])

    print(np.array([1, 2, 3, 4, 5, 6]).reshape((2, 3)))

    a = np.array([[1, 0, 1], [0, 1, 0]])
    x = np.array([1, 2, 3])
    y = np.array([1, 0, 2])
    print(np.dot(x, y))  # 7
    print(a.dot(x))  # [4 2]

    e = np.eye(4, 5)
    print(e)
    print(e[[0]])
    print(e[[0, 3]])
    print(e[[0, 3], [1, 3]])

    return


def matplotlib_test():
    x = np.array([1, 2, 3, 4, 5])
    y = x**2
    plt.plot(x, y)
    plt.show()


def homework1():
    word_ids = {}
    vectors = []
    lines = []

    # fill bag of words matrix

    with open(utils.PATH.COURSE_FILE(1, 'sentences.txt')) as file:
        lines = [line.strip() for line in map(str.lower, file.readlines())]
        n = 0
        for line in lines:
            vector = [0]*n
            tokens = list(filter(None, re.split('[^a-z]', line)))
            counts = Counter(tokens)
            for t_key in counts:
                t_value = counts[t_key]
                if t_key in word_ids:
                    word_id = word_ids[t_key]
                    vector[word_id] = t_value
                else:
                    word_ids[t_key] = n
                    vector.append(t_value)
                    n += 1
            vectors.append(vector)

        for vector in vectors:
            vector += [0]*(n - len(vector))

    bof = np.array(vectors)
    print(bof.shape)

    first = bof[0, :]
    for i in range(bof.shape[0]):
        vector = bof[i, :]
        dist = cosine(first, vector)
        print(i, '\t', round(dist, 5), '\t', lines[i])


    return


def homework2():

    def f(x):
        return np.sin(x/5.0)*np.exp(x/10.0) + 5.0*np.exp(-x/2.0)

    def pol(x, w):
        n = len(w)
        m = len(x)
        xs = np.zeros((m, n))
        p = np.ones((m,))
        for i in range(n):
            xs[:, i] = p
            p *= x
        return xs.dot(w)

    data = [np.array([1.0, 15.0]),
            np.array([1.0, 8.0, 15.0]),
            np.array([1.0, 4.0, 10.0, 15.0])]

    x_new = np.arange(1.0, 15.0, 0.1)
    f_new = f(x_new)

    for xs in data:
        y = f(xs)
        n = len(xs)
        A = np.zeros((n, n))
        for i in range(n):
            x = xs[i]
            p = 1
            for j in range(n):
                A[i, j] = p
                p *= x
        w = np.linalg.solve(A, y)
        print(w)
        p_new = pol(x_new, w)
        plt.plot(x_new, f_new, 'o', x_new, p_new, '-')
        plt.show()

    return