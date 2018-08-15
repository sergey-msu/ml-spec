import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def header():
    return 'WEEK 3: Natural Language Processing';


def run():

    df = pd.read_csv(utils.PATH.COURSE_FILE(5, 'SMSSpamCollection.txt', 'week3'), sep='\t', header=None)
    df.replace({ 0: { 'spam': 1, 'ham': 0 } }, inplace=True)
    print(df.head())

    X = df[1].values
    y = df[0].values

    # Q1

    def get_model(vectorizer, estimator):
        return Pipeline(steps =
            [('vect', vectorizer),
             ('estimator', estimator)
            ])

    model = get_model(CountVectorizer(), LogisticRegression())

    cv = KFold(n_splits=10, shuffle=True, random_state=2)
    score = cross_val_score(model, X, y, cv=cv, scoring='f1').mean()
    print(score)

    utils.PATH.SAVE_RESULT((5, 3), (1, 1), round(score, 1))


    # Q2

    model.fit(X, y)

    msgs =[
        "FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! Subscribe6GB",
        "FreeMsg: Txt: claim your reward of 3 hours talk time",
        "Have you visited the last lecture on physics?",
        "Have you visited the last lecture on physics? Just buy this book and you will have all materials! Only 99$",
        "Only 99$"]
    results = []

    for msg in msgs:
        result = model.predict([msg])[0]
        results.append(result)

    utils.PATH.SAVE_RESULT((5, 3), (1, 2), results)

    # Q3

    model1 = get_model(CountVectorizer(ngram_range=(2, 2)), LogisticRegression())
    model2 = get_model(CountVectorizer(ngram_range=(3, 3)), LogisticRegression())
    model3 = get_model(CountVectorizer(ngram_range=(1, 3)), LogisticRegression())
    score1 = cross_val_score(model1, X, y, cv=cv, scoring='f1').mean()
    score2 = cross_val_score(model2, X, y, cv=cv, scoring='f1').mean()
    score3 = cross_val_score(model3, X, y, cv=cv, scoring='f1').mean()
    res = np.round(np.array([score1, score2, score3]), 2)
    print(np.round(np.array([score1, score2, score3]), 2))

    utils.PATH.SAVE_RESULT((5, 3), (1, 3), res)

    # Q4

    vect1 = CountVectorizer(ngram_range=(2, 2))
    X1 = vect1.fit_transform(X)
    model1 = MultinomialNB()
    score1 = cross_val_score(model1, X1, y, cv=cv, scoring='f1').mean()

    vect2 = CountVectorizer(ngram_range=(3, 3))
    X2 = vect2.fit_transform(X)
    model2 = MultinomialNB()
    score2 = cross_val_score(model2, X2, y, cv=cv, scoring='f1').mean()

    vect3 = CountVectorizer(ngram_range=(1, 3))
    X3 = vect3.fit_transform(X)
    model3 = MultinomialNB()
    score3 = cross_val_score(model3, X3, y, cv=cv, scoring='f1').mean()

    res = np.round(np.array([score1, score2, score3]), 2)
    print(np.round(np.array([score1, score2, score3]), 2))

    utils.PATH.SAVE_RESULT((5, 3), (1, 4), res)

    # Q5


    vect = TfidfVectorizer(ngram_range=(1, 1))
    X = vect.fit_transform(X)
    model = LogisticRegression()
    score = cross_val_score(model, X, y, cv=cv, scoring='f1').mean()
    print(score)

    return

