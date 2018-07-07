#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import StratifiedKFold


ABS_PATH = os.path.dirname(os.path.abspath("__file__"))
DATASETS_DIR = ABS_PATH + "/datasets/{}"
DATASET_PATH = DATASETS_DIR.format('tfidf_1.csv')


dataset = pd.read_csv(DATASET_PATH)
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, -2].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()'
y = le.fit_transform(y)

first_layer_units = len(X[0])

kfold = StratifiedKFold(n_splits=10, shuffle=True)
cvscores = []

for train, test in kfold.split(X, y):
    classifier = Sequential()
    classifier.add(Dense(units=first_layer_units, activation='relu'))
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dropout(.3))
    classifier.add(Dense(units=3, activation='softmax'))
    classifier.compile(optimizer='adam', metrics=['accuracy'],
                       loss='sparse_categorical_crossentropy')
    
    scores = classifier.fit(X[train], y[train], 
                            validation_data=(X[test], y[test]),
                            epochs=100)
    cvscores.append(scores.history['val_acc'][-1])

print(cvscores)
print(np.array(cvscores).mean())