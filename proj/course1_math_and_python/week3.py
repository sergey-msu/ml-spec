import math
import numpy as np
import utils
from scipy.optimize import minimize, differential_evolution
from matplotlib import pyplot as plt


def header():
    return 'WEEK 3: Optimization and Matrix Decomposition';


def run():

    homework1()

    return


def homework1():

    def f(x):
        return np.sin(x/5.0)*np.exp(x/10.0) + 5.0*np.exp(-x/2.0)

    res = minimize(f, 30.0, method='BFGS')
    print(res)
    print('-------------------')

    res = differential_evolution(f, [(1, 30)])
    print(res)
    print('-------------------')

    def h(x):
        return f(x).astype(int)

    xs = np.arange(1, 30, 0.1)
    fs = f(xs)
    hs = h(xs)

    plt.plot(xs, fs, 'o', xs, hs, '-')
    #plt.show()

    res = minimize(h, 30.0, method='BFGS')
    print(res)
    print('-------------------')

    res = differential_evolution(h, [(1, 30)])
    print(res)
    print('-------------------')

    return