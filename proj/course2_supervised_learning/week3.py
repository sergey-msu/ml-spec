import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score


def header():
    return 'WEEK 3: Linear Models';


def run():

    homework()

    return


def homework():

    df = pd.read_csv(utils.PATH.COURSE_FILE(2, 'data.csv'))
    print(df.shape)
    #print(df.head())
    #print(df.info())

    X = df.drop('Grant.Status', axis=1)
    y = df['Grant.Status']

    numeric_cols = ['RFCD.Percentage.1', 'RFCD.Percentage.2', 'RFCD.Percentage.3',
                    'RFCD.Percentage.4', 'RFCD.Percentage.5',
                    'SEO.Percentage.1', 'SEO.Percentage.2', 'SEO.Percentage.3',
                    'SEO.Percentage.4', 'SEO.Percentage.5',
                    'Year.of.Birth.1', 'Number.of.Successful.Grant.1', 'Number.of.Unsuccessful.Grant.1']
    categorical_cols = list(set(X.columns.values.tolist()) - set(numeric_cols))

    X_real_zeros = X[numeric_cols].fillna(0.0)
    X_real_means = X[numeric_cols].fillna(X.mean())
    X_cat        = X[categorical_cols].fillna('NA').applymap(str)

    encoder = DictVectorizer(sparse=False)

    X_cat_oh = encoder.fit_transform(X_cat.T.to_dict().values())
    print(X_cat_oh.shape)

    X_train_real_zeros, X_test_real_zeros, \
    y_train, y_test = train_test_split(X_real_zeros, y, test_size=0.3, random_state=0)

    X_train_real_means, X_test_real_means, \
    y_train, y_test = train_test_split(X_real_means, y, test_size=0.3, random_state=0)

    X_train_cat_oh, X_test_cat_oh, \
    y_train, y_test = train_test_split(X_cat_oh, y, test_size=0.3, random_state=0)

    X_train_zeros = np.hstack([X_train_real_zeros.values, X_train_cat_oh])
    X_test_zeros  = np.hstack([X_test_real_zeros.values, X_test_cat_oh])
    X_train_means = np.hstack([X_train_real_means.values, X_train_cat_oh])
    X_test_means  = np.hstack([X_test_real_means.values, X_test_cat_oh])

    def task_1():
        alg = LogisticRegression()
        params_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
        grid = GridSearchCV(alg, params_grid, cv=3, n_jobs=-1)
        grid.fit(X_train_zeros, y_train)
        print(grid.best_params_, grid.best_score_)

        y_pred = grid.predict_proba(X_test_zeros)[:, 1]
        auc_1 = roc_auc_score(y_test, y_pred)
        print('ROC AUC on zeroes:', auc_1)
        #plot_scores(grid)

        grid.fit(X_train_means, y_train)
        print(grid.best_params_, grid.best_score_)

        y_pred = grid.predict_proba(X_test_means)[:, 1]
        auc_2 = roc_auc_score(y_test, y_pred)
        print('ROC AUC on zeroes:', auc_2)
        #plot_scores(grid)

        write_answer_1(auc_2, auc_1)
    #task_1()


    ##### Scaling #####

    scaler = StandardScaler()
    X_train_real_scaled = scaler.fit_transform(X_train_real_zeros)
    X_test_real_scaled  = scaler.transform(X_test_real_zeros)

    #data_numeric = pd.DataFrame(X_train_real_scaled, columns=numeric_cols)
    #list_cols = ['Number.of.Successful.Grant.1', 'SEO.Percentage.2', 'Year.of.Birth.1']
    #scatter_matrix(data_numeric[list_cols], alpha=0.5, figsize=(10, 10))
    #plt.show()

    X_train_scaled = np.hstack([X_train_real_scaled, X_train_cat_oh])
    X_test_scaled  = np.hstack([X_test_real_scaled, X_test_cat_oh])

    def task_2():
        alg = LogisticRegression()
        params_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
        grid = GridSearchCV(alg, params_grid, cv=3, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        print(grid.best_params_, grid.best_score_)

        y_pred = grid.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        print('ROC AUC on scaled zeroes:', auc)
        #plot_scores(grid)

        write_answer_2(auc_2)

        return
    #task_2()

    def example():
        np.random.seed(0)

        param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
        cv = 3
        """Сэмплируем данные из первой гауссианы"""
        data_0 = np.random.multivariate_normal([0,0], [[0.5,0],[0,0.5]], size=40)
        """И из второй"""
        data_1 = np.random.multivariate_normal([0,1], [[0.5,0],[0,0.5]], size=40)
        """На обучение берём 20 объектов из первого класса и 10 из второго"""
        example_data_train = np.vstack([data_0[:20,:], data_1[:10,:]])
        example_labels_train = np.concatenate([np.zeros((20)), np.ones((10))])
        """На тест - 20 из первого и 30 из второго"""
        example_data_test = np.vstack([data_0[20:,:], data_1[10:,:]])
        example_labels_test = np.concatenate([np.zeros((20)), np.ones((30))])
        """Задаём координатную сетку, на которой будем вычислять область классификации"""
        xx, yy = np.meshgrid(np.arange(-3, 3, 0.02), np.arange(-3, 3, 0.02))
        """Обучаем регрессию без балансировки по классам"""
        optimizer = GridSearchCV(LogisticRegression(), param_grid, cv=cv, n_jobs=-1)
        optimizer.fit(example_data_train, example_labels_train)
        """Строим предсказания регрессии для сетки"""
        Z = optimizer.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
        plt.scatter(data_0[:,0], data_0[:,1], color='red')
        plt.scatter(data_1[:,0], data_1[:,1], color='blue')
        """Считаем AUC"""
        auc_wo_class_weights = roc_auc_score(example_labels_test, optimizer.predict_proba(example_data_test)[:,1])
        plt.title('Without class weights')
        plt.show()
        print('AUC: %f'%auc_wo_class_weights)
        """Для второй регрессии в LogisticRegression передаём параметр class_weight='balanced'"""
        optimizer = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=cv, n_jobs=-1)
        optimizer.fit(example_data_train, example_labels_train)
        Z = optimizer.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
        plt.scatter(data_0[:,0], data_0[:,1], color='red')
        plt.scatter(data_1[:,0], data_1[:,1], color='blue')
        auc_w_class_weights = roc_auc_score(example_labels_test, optimizer.predict_proba(example_data_test)[:,1])
        plt.title('With class weights')
        plt.show()
        print('AUC: %f'%auc_w_class_weights)
    #example()

    def task_3():
        alg = LogisticRegression(class_weight='balanced')
        params_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
        grid = GridSearchCV(alg, params_grid, cv=3, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        print(grid.best_params_, grid.best_score_)

        y_pred = grid.predict_proba(X_test_scaled)[:, 1]
        auc_1 = roc_auc_score(y_test, y_pred)
        print('ROC AUC on scaled zeroes:', auc_1)
        #plot_scores(grid)

        ## Balanced

        np.random.seed(0)
        n0 = sum(y_train==0)
        n1 = sum(y_train==1)
        print(n0, n1)

        y_less = np.nonzero(y_train)[0]
        indices_to_add = y_less[np.random.randint(0, len(y_less), n0 - n1)]
        X_train_to_add = X_train_scaled[indices_to_add, :]
        y_train_to_add = y_train.values[indices_to_add]
        X_train_balanced = np.vstack([X_train_scaled, X_train_to_add])
        y_train_balanced = np.hstack([y_train, y_train_to_add])

        alg = LogisticRegression()
        params_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
        grid = GridSearchCV(alg, params_grid, cv=3, n_jobs=-1)
        grid.fit(X_train_balanced, y_train_balanced)
        print(grid.best_params_, grid.best_score_)

        y_pred = grid.predict_proba(X_test_scaled)[:, 1]
        auc_2 = roc_auc_score(y_test, y_pred)
        print('ROC AUC on scaled balanced zeroes:', auc_2)

        write_answer_3(auc_1, auc_2)

        return
    #task_3()


    X_train_real_zeros, X_test_real_zeros, \
    y_train, y_test = train_test_split(X_real_zeros, y, test_size=0.3, random_state=0, stratify=y)

    X_train_cat_oh, X_test_cat_oh, \
    y_train, y_test = train_test_split(X_cat_oh, y, test_size=0.3, random_state=0, stratify=y)

    scaler = StandardScaler()
    X_train_real_scaled = scaler.fit_transform(X_train_real_zeros)
    X_test_real_scaled  = scaler.transform(X_test_real_zeros)

    X_train_scaled = np.hstack([X_train_real_scaled, X_train_cat_oh])
    X_test_scaled  = np.hstack([X_test_real_scaled, X_test_cat_oh])

    def task_4():
        alg = LogisticRegression(class_weight='balanced')
        params_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
        grid = GridSearchCV(alg, params_grid, cv=3, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        print(grid.best_params_, grid.best_score_)

        y_pred = grid.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        print('ROC AUC on scaled y-stratified zeroes:', auc)

        write_answer_4(auc)

        return
    #task_4()

    def example_2():
        param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
        cv = 3

        """Инициализируем класс, который выполняет преобразование"""
        transform = PolynomialFeatures(2)
        """Сэмплируем данные из первой гауссианы"""
        data_0 = np.random.multivariate_normal([0,0], [[0.5,0],[0,0.5]], size=40)
        """И из второй"""
        data_1 = np.random.multivariate_normal([0,1], [[0.5,0],[0,0.5]], size=40)
        """На обучение берём 20 объектов из первого класса и 10 из второго"""
        example_data_train = np.vstack([data_0[:20,:], data_1[:10,:]])
        example_labels_train = np.concatenate([np.zeros((20)), np.ones((10))])
        """На тест - 20 из первого и 30 из второго"""
        example_data_test = np.vstack([data_0[20:,:], data_1[10:,:]])
        """Обучаем преобразование на обучающей выборке, применяем его к тестовой"""
        example_data_train_poly = transform.fit_transform(example_data_train)
        example_data_test_poly = transform.transform(example_data_test)
        example_labels_test = np.concatenate([np.zeros((20)), np.ones((30))])
        """Обращаем внимание на параметр fit_intercept=False"""
        optimizer = GridSearchCV(LogisticRegression(class_weight='balanced', fit_intercept=False), param_grid, cv=cv, n_jobs=-1)
        optimizer.fit(example_data_train_poly, example_labels_train)
        """Задаём координатную сетку, на которой будем вычислять область классификации"""
        xx, yy = np.meshgrid(np.arange(-3, 3, 0.02), np.arange(-3, 3, 0.02))
        Z = optimizer.predict(transform.transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
        plt.scatter(data_0[:,0], data_0[:,1], color='red')
        plt.scatter(data_1[:,0], data_1[:,1], color='blue')
        plt.title('With class weights')
        plt.show()


        transform = PolynomialFeatures(15)
        example_data_train_poly = transform.fit_transform(example_data_train)
        example_data_test_poly = transform.transform(example_data_test)
        optimizer = GridSearchCV(LogisticRegression(class_weight='balanced', fit_intercept=False), param_grid, cv=cv, n_jobs=-1)
        optimizer.fit(example_data_train_poly, example_labels_train)
        Z = optimizer.predict(transform.transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
        plt.scatter(data_0[:,0], data_0[:,1], color='red')
        plt.scatter(data_1[:,0], data_1[:,1], color='blue')
        plt.title('Corrected class weights')
        plt.show()
        return
    #example_2()


    poly = PolynomialFeatures(2)
    X_train_real_zeros_poly = poly.fit_transform(X_train_real_zeros)
    X_test_real_zeros = poly.transform(X_test_real_zeros)

    scaler = StandardScaler()
    X_train_real_poly_scaled = scaler.fit_transform(X_train_real_zeros_poly)
    X_test_real_poly_scaled  = scaler.transform(X_test_real_zeros)

    X_train_poly_scaled = np.hstack([X_train_real_poly_scaled, X_train_cat_oh])
    X_test_poly_scaled  = np.hstack([X_test_real_poly_scaled, X_test_cat_oh])

    def task_5():
        alg = LogisticRegression(class_weight='balanced', fit_intercept=False)
        params_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
        grid = GridSearchCV(alg, params_grid, cv=3, n_jobs=-1)
        grid.fit(X_train_poly_scaled, y_train)
        print(grid.best_params_, grid.best_score_)

        y_pred = grid.predict_proba(X_test_poly_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        print('ROC AUC on scaled y-stratified zeroes:', auc)

        write_answer_5(auc)
        return
    #task_5()

    def task_6():
        alg = LogisticRegression(class_weight='balanced', penalty='l1')
        params_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
        grid = GridSearchCV(alg, params_grid, cv=3, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        print(grid.best_params_, grid.best_score_)

        alg = grid.best_estimator_
        zero_ids = np.where(alg.coef_[0,:X_train_real_scaled.shape[1]] == 0)[0]

        write_answer_6(zero_ids)
        return
    task_6()

    return


def plot_scores(optimizer):
    scores = [[item[0]['C'],
               item[1],
               (np.sum((item[2]-item[1])**2)/(item[2].size-1))**0.5] for item in optimizer.grid_scores_]
    scores = np.array(scores)
    plt.semilogx(scores[:,0], scores[:,1])
    plt.fill_between(scores[:,0], scores[:,1]-scores[:,2],
                                  scores[:,1]+scores[:,2], alpha=0.3)
    plt.show()


def write_answer_1(auc_1, auc_2):
    auc = (auc_1 + auc_2)/2
    with open(utils.PATH.STORE_FOR(2, 'preprocessing_lr_answer1.txt'), 'w') as fout:
        fout.write(str(auc))


def write_answer_2(auc):
    with open(utils.PATH.STORE_FOR(2, 'preprocessing_lr_answer2.txt'), 'w') as fout:
        fout.write(str(auc))


def write_answer_3(auc_1, auc_2):
    auc = (auc_1 + auc_2) / 2
    with open(utils.PATH.STORE_FOR(2, 'preprocessing_lr_answer3.txt'), 'w') as fout:
        fout.write(str(auc))


def write_answer_4(auc):
    with open(utils.PATH.STORE_FOR(2, 'preprocessing_lr_answer4.txt'), 'w') as fout:
        fout.write(str(auc))


def write_answer_5(auc):
    with open(utils.PATH.STORE_FOR(2, 'preprocessing_lr_answer5.txt'), 'w') as fout:
        fout.write(str(auc))


def write_answer_6(features):
    with open(utils.PATH.STORE_FOR(2, 'preprocessing_lr_answer6.txt'), 'w') as fout:
        fout.write(' '.join([str(num) for num in features]))