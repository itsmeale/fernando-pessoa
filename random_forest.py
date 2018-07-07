#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os


ABS_PATH = os.path.dirname(os.path.abspath("__file__"))
DATASETS_DIR = ABS_PATH + "/datasets/{}"
DATASET_PATH = DATASETS_DIR.format('tfidf_1.csv')

dataset = pd.read_csv(DATASET_PATH)
X = dataset.iloc[:,:-2].values
y = dataset.iloc[:,-2].values

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

estimators = []
for i in range(500):
    classifier = RandomForestClassifier(n_estimators=i+1)
    somatorio = 0
    for j in range(10):
        accs = cross_val_score(estimator=classifier, X=X, y=y, cv=10)
        somatorio += accs.mean()
    estimators.append((i, somatorio/10))
    print((i, somatorio/10))
