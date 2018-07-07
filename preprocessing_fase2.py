#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import os


# caminho absoluto da pasta de onde preprocessamento esta sendo executado
ABS_PATH = os.path.dirname(os.path.abspath("__file__"))
DATASETS_DIR = ABS_PATH + "/datasets/{}"
BASE_DATASET_PATH = DATASETS_DIR.format('dataset.csv')


# le o dataset previamente preprocessado
base_dataset = pd.read_csv(BASE_DATASET_PATH)

# dicionario de configuracao de preprocessanento
confs = {
        'bow_1':CountVectorizer(binary=True),
        'bow_2':CountVectorizer(binary=True, ngram_range=(2,2)),
        'bow_3':CountVectorizer(binary=True, ngram_range=(3,3)),
        'tfidf_1':TfidfVectorizer(),
        'tfidf_2':TfidfVectorizer(ngram_range=(2,2)),
        'tfidf_3':TfidfVectorizer(ngram_range=(3,3))
}

# gera um dataset para cada configuracao de preprocessamento
for conf in confs:
    vectorizer = confs[conf]
    
    X = vectorizer.fit_transform(base_dataset['poema'])
    features = vectorizer.get_feature_names()
    X = X.toarray()
    
    new_dataset = pd.DataFrame(X, columns=features)
    
    to_delete = []
    for i, feature in enumerate(features):
        soma = sum([1 for value in X[X[:,i] != 0]])
        max_cut = len(X)*.33
        min_cut = len(X)*.03
        
        if soma > max_cut or soma < min_cut:
            to_delete.append(feature)
    
    new_dataset = new_dataset.drop(to_delete, axis=1)
    
    new_dataset["$heteronomio"] = base_dataset["heteronomio"]
    new_dataset["$poema"] = base_dataset["poema"]
    new_dataset_name = conf+'.csv'
    new_dataset.to_csv(DATASETS_DIR.format(new_dataset_name), index=False)
    print(new_dataset_name + ' finished...')
