import math
import numpy as np
import pandas as pd
import utils
import scipy.stats as sts
from collections import Counter
from matplotlib import pyplot as plt


def header():
    return 'WEEK 4: Probabiliy and Statistics';


def run():

    #samples()
    homework()

    return


def samples():

    n = 100
    sample = np.random.choice([1, 2, 3, 4, 5, 6], n)
    c = Counter(sample)
    print(c)

    freqs = {k: v/n for k, v in c.items()}
    print(freqs)

    print('----------')

    n = 100
    norm_rv = sts.norm(0, 1)
    sample = norm_rv.rvs(n)

    x = np.linspace(-4, 4, 100)
    cdf = norm_rv.cdf(x)
    plt.plot(x, cdf, label='Theoretical CDF')

    ecdf = ECDF(sample)
    plt.step(ecdf.x, ecdf.y, label='ECDF')
    plt.show()

    plt.hist(sample, bins=10, normed=True)
    plt.ylabel('fraction of samples')
    plt.xlabel('$x$')
    plt.show()

    print('----------')

    df = pd.DataFrame(sample, columns=['KDE'])
    ax = df.plot(kind='density')

    x = np.linspace(-4, 4, 100)
    pdf = norm_rv.pdf(x)
    plt.plot(x, pdf, label='theoretical pdf', alpha=0.5)
    plt.legend()
    plt.ylabel('$f(x)$')
    plt.xlabel('$x$')
    plt.show()

    return


def homework():

    k = 3
    mean = k
    var  = 2*k
    chi2_rv = sts.chi2(df=k)

    n = 1000
    sample = chi2_rv.rvs(n)
    #print(sample)

    x = np.linspace(0, 15, 1000)
    y = chi2_rv.pdf(x)
    plt.hist(sample, bins=30, normed=True)
    plt.plot(x, y)
    plt.ylabel('$\chi^2_k, k=3$')
    plt.xlabel('$x$')
    plt.show()

    ns = [5, 10, 50, 100, 1000]
    N = 1000
    for n in ns:
        samples = chi2_rv.rvs((N, n))
        means = np.mean(samples, axis=1)

        s_mean = mean
        s_var  = var/n

        print('Due to Central Limit Theorem:')
        print('Sample mean tends to'+str(s_mean))
        print('Sample variance tends to'+str(s_var))

        norm_rv = sts.norm(s_mean, math.sqrt(s_var))
        y_norm = norm_rv.pdf(x)

        plt.plot(x, y_norm)
        plt.hist(means, bins=30, normed=True)
        plt.ylabel('$N({0},{1})$'.format(s_mean, s_var))
        plt.xlabel('$x$')
        plt.show()

    return