import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils


def header():
    return 'WEEK 1: Intro and Linear Models';


def run():

    #notebook()
    homework()

    return

def notebook():
    data = pd.read_csv(utils.PATH.COURSE_FILE(2, 'weights_heights.csv'), index_col='Index')

    data.plot(y='Height', kind='hist', color='red',  title='Height (inch.) distribution')
    print(data.head(5))
    plt.show()

    data.plot(y='Weight', kind='hist', color='red',  title='Weight (inch.) distribution')
    plt.show()

    def make_bmi(height_inch, weight_pound):
        METER_TO_INCH, KILO_TO_POUND = 39.37, 2.20462
        return (weight_pound / KILO_TO_POUND) /            (height_inch / METER_TO_INCH) ** 2

    data['BMI'] = data.apply(lambda row: make_bmi(row['Height'],
                                                  row['Weight']), axis=1)

    sns.pairplot(data)
    plt.show()

    def weight_category(weight):
        if weight < 120: return 1
        if weight >= 150: return 3
        return 2

    data['weight_cat'] = data['Weight'].apply(weight_category)

    sns.boxplot(x="weight_cat", y="Height", data=data[["weight_cat", "Height"]])
    plt.show()

    data.plot(x="Weight", y="Height", kind="scatter", title="Height-Weight dependence")
    plt.show()

    def error(w0, w1):
        sum = 0
        for xi, yi in zip(data['Weight'], data['Height']):
            sum += (yi - (w0 + w1*xi))**2
        return sum

    x = np.linspace(70, 180, 2)
    y1 = 60 + x*0.05
    y2 = 50 + x*0.16
    plt.plot(x, y1, color='red')
    plt.plot(x, y2, color='orange')
    plt.scatter(data['Weight'], data['Height'])
    plt.xlabel('Weight')
    plt.ylabel('Height')
    plt.show()


    n = 10
    w1 = np.linspace(0, 2, n)
    w0 = np.array([50]*n)
    err = []
    for _w0, _w1 in zip(w0, w1):
        e = error(_w0, _w1)
        err.append(e)
    plt.plot(w1, err)
    plt.xlabel('w1')
    plt.ylabel('error')
    plt.title('Error function')
    plt.show()


    from scipy import optimize

    def error_w1(w1):
        return error(50, w1)

    w_opt = optimize.minimize_scalar(error_w1, bounds=(-5, 5))
    w1_opt = w_opt.x
    print(w1_opt)


    x = np.linspace(70, 180, 2)
    y = 50 + x*w1_opt
    plt.plot(x, y, color='red')
    plt.scatter(data['Weight'], data['Height'], color='green')
    plt.xlabel('Weight')
    plt.ylabel('Height')
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D


    fig = plt.figure()
    ax = fig.gca(projection='3d') # get current axis

    # Создаем массивы NumPy с координатами точек по осям X и У.
    # Используем метод meshgrid, при котором по векторам координат
    # создается матрица координат. Задаем нужную функцию Z(x, y).
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    # Наконец, используем метод *plot_surface* объекта
    # типа Axes3DSubplot. Также подписываем оси.
    surf = ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    W0 = np.arange(45, 55, 0.1)
    W1 = np.arange(0, 1, 0.01)
    W0, W1 = np.meshgrid(W0, W1)
    ERR = error(W0, W1)

    surf = ax.plot_surface(W0, W1, ERR)
    ax.set_xlabel('Intercept')
    ax.set_ylabel('Slope')
    ax.set_zlabel('Error')
    plt.show()

    def error_vec(w):
        return error(w[0], w[1])

    res = optimize.minimize(error_vec, x0=[0, 0], bounds=[(-100, 100), (-5, 5)])
    w0_opt = res.x[0]
    w1_opt = res.x[1]

    x = np.linspace(70, 180, 2)
    y = w0_opt + x*w1_opt
    plt.plot(x, y, color='red')
    plt.scatter(data['Weight'], data['Height'], color='green')
    plt.xlabel('Weight')
    plt.ylabel('Height')
    plt.show()

    return


def homework():

    df = pd.read_csv(utils.PATH.COURSE_FILE(2, 'advertising.csv'))
    print(df.head(5))
    print(df.info())

    X = df[['TV', 'Radio', 'Newspaper' ]].values
    y = df['Sales'].values

    X = (X - X.mean(axis=0))/X.std(axis=0)   # scale
    X = np.hstack((np.ones([len(y), 1]), X)) # add column of 1-s

    # Q1
    sales_med = np.median(y)
    answer1 = mserror(y, [sales_med]*len(y))
    print(answer1)
    write_answer_to_file(answer1, '1.txt')

    # Q2
    norm_eq_weights = normal_equation(X, y)
    print(norm_eq_weights)

    X_test = np.array([[1.0, 0.0, 0.0, 0.0]])
    answer2 = X_test.dot(norm_eq_weights)[0]
    write_answer_to_file(answer2, '2.txt')

    # Q3
    y_pred = linear_prediction(X, norm_eq_weights)
    answer3 = mserror(y, y_pred)
    write_answer_to_file(answer3, '3.txt')

    # Q4
    iter_num = 1e5
    w_init = np.zeros_like(norm_eq_weights)
    w, errors = stochastic_gradient_descent(X, y, w_init)

    print(w)

    plt.plot(range(50), errors[:50])
    plt.show()

    y_pred = linear_prediction(X, w)
    answer4 = mserror(y, y_pred)
    write_answer_to_file(answer4, '4.txt')

    return


def mserror(y, y_pred):
    return ((y - y_pred)**2).mean()


def normal_equation(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def linear_prediction(X, w):
    return X.dot(w)


def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    grad = 2*(X[train_ind, :].dot(w) - y[train_ind])*X[train_ind, :]/len(y)
    return w - eta*grad


def stochastic_gradient_descent(X, y, w_init,
                                eta=0.01, max_iter=1e4, min_weight_dist=1e-8, seed=42, verbose=False):
    weight_dist = np.inf
    w = w_init
    errors = []
    iter_num = 0

    np.random.seed(seed)

    while (weight_dist > min_weight_dist and iter_num < max_iter):
        random_ind = np.random.randint(X.shape[0])
        w_new = stochastic_gradient_step(X, y, w, random_ind, eta)
        weight_dist = np.linalg.norm(w - w_new)
        w = w_new
        error = mserror(y, linear_prediction(X, w))
        errors.append(error)

    return w, errors


def write_answer_to_file(answer, filename):
    with open(utils.PATH.STORE_FOR(2, filename), 'w') as file:
        file.write(str(round(answer, 3)))
