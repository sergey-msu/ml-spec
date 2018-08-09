import os
import copy
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
from gensim import corpora, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import artm


def header():
    return 'WEEK 4: Thematic Modeling';


def run():

    #homework()

    return


def homework():
    np.random.seed(76543)

    with open(utils.PATH.COURSE_FILE(3, 'week4//recipes.json')) as file:
        recipes = json.load(file)

    docs = [recipe['ingredients'] for recipe in recipes]
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    print(docs[0])
    print(corpus[0])

    fpath = utils.PATH.STORE_FOR(3, 'lda.model', 'week4')
    if os.path.exists(fpath):
        lda = models.LdaModel.load(fpath)
    else:
        lda = models.LdaModel(corpus, num_topics=40, passes=5)
        lda.save(fpath)

    topics = lda.show_topics(num_topics=40, num_words=10, formatted=False)

    # Q1

    answer = []
    ingreds = ["salt", "sugar", "water", "mushrooms", "chicken", "eggs"]
    for ingred in ingreds:
        id = str(dictionary.token2id[ingred])
        cnt = 0
        for topic in topics:
            cnt += len(list(filter(lambda w: w[0]==id, topic[1])))
        answer.append(cnt)
        print(ingred, cnt)

    utils.PATH.SAVE_RESULT((3, 4), (1, 1), answer)

    # Q2

    dictionary2 = copy.deepcopy(dictionary)

    often_ids = list(filter(lambda k: dictionary2.dfs[k]>4000, dictionary2.dfs))
    print(often_ids)

    dictionary2.filter_tokens(often_ids)
    dict_size_before = len(dictionary)
    dict_size_after = len(dictionary2)

    corpus2 = [dictionary2.doc2bow(doc) for doc in docs]

    def corp_len(corp):
        s = 0
        for c in corp:
            s += len(c)
        return s

    corpus_size_before = corp_len(corpus)
    corpus_size_after = corp_len(corpus2)

    answer = [dict_size_before, dict_size_after, corpus_size_before, corpus_size_after]
    print(answer)
    utils.PATH.SAVE_RESULT((3, 4), (1, 2), answer)

    # Q3

    fpath = utils.PATH.STORE_FOR(3, 'lda.model2', 'week4')
    if os.path.exists(fpath):
        lda2 = models.LdaModel.load(fpath)
    else:
        lda2 = models.LdaModel(corpus2, num_topics=40, passes=5)
        lda2.save(fpath)

    #coherence = lda.top_topics(corpus)
    #mean_coherence = np.mean([theme_coherence[1] for theme_coherence in coherence])
    #coherence2 = lda2.top_topics(corpus2)
    #mean_coherence2 = np.mean([theme_coherence2[1] for theme_coherence2 in coherence2])
    #
    #answer = [mean_coherence, mean_coherence2]
    #print(answer)
    #utils.PATH.SAVE_RESULT((3, 4), (1, 3), answer)

    # Q4

    #print(lda2.get_document_topics(corpus2[0]))
    #print(lda2.alpha)
    #
    #def calc_sparsity(model, corp):
    #    result = 0
    #    for c in corp:
    #        topics = model.get_document_topics(c)
    #        for topic in topics:
    #            if topic[1] > 0.01:
    #                result += 1
    #    return result
    #
    #count_model2 = calc_sparsity(lda2, corpus2)
    #
    #fpath = utils.PATH.STORE_FOR(3, 'lda.model3', 'week4')
    #if os.path.exists(fpath):
    #    lda3 = models.LdaModel.load(fpath)
    #else:
    #    lda3 = models.LdaModel(corpus2, num_topics=40, passes=5, alpha=1.0)
    #    lda3.save(fpath)
    #
    #print(lda3.get_document_topics(corpus2[0]))
    #print(lda3.alpha)
    #
    #count_model3 = calc_sparsity(lda3, corpus2)
    #
    #answer = [count_model2, count_model3]
    #
    #utils.PATH.SAVE_RESULT((3, 4), (1, 4), answer)

    # Q5

    cuisines = list(set([recipe['cuisine'] for recipe in recipes]))
    cuisines_map = dict((cuisine, i) for i, cuisine in enumerate(cuisines))
    print(cuisines_map)

    X = np.zeros((len(corpus2), 40))
    y = np.zeros((len(corpus2),))
    i = 0
    for c in corpus2:
        topics = lda2.get_document_topics(c)
        for topic in topics:
            X[i, topic[0]] = topic[1]
        y[i] = cuisines_map[recipes[i]['cuisine']]
        i += 1

    alg = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    score = cross_val_score(alg, X, y, cv=3, n_jobs=-1)
    answer = score.mean()

    utils.PATH.SAVE_RESULT((3, 4), (1, 5), answer)

    return