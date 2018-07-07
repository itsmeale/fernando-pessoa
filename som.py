#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np

ABS_PATH = os.path.dirname(os.path.abspath("__file__"))
DATASETS_DIR = ABS_PATH + "/datasets/{}"
DATASET_PATH = DATASETS_DIR.format('tfidf_1.csv')

dataset = pd.read_csv(DATASET_PATH)
X = dataset.iloc[:,:-2].values
y = dataset.iloc[:,-2].values

import sompy
mapsize = [20, 20]
som = sompy.SOMFactory.build(X, mapsize, mask=None, mapshape='planar', 
                             initialization='random', 
                             neighborhood='gaussian',
                             training='batch')
som.train(n_job=3, verbose='info')
topographic_error = som.calculate_topographic_error()
quantization_error = np.mean(som._bmu[1])
print ("Distorção topológica = {0}\nErro de quantização = {1}"
       .format(topographic_error, quantization_error))

u = sompy.umatrix.UMatrixView(10, 10, 
                              'umatrix', 
                              show_axis=True, 
                              text_size=8, 
                              show_text=True)
UMAT = u.build_u_matrix(som, distance=0, row_normalized=False)
UMAT = u.show(som, distance2=1, row_normalized=False, 
              show_data=False, contooor=True, blob=False)

number_of_clusters = 3
som.cluster(number_of_clusters)

h = sompy.hitmap.HitMapView(10, 10, 'Hitmap View', text_size=8, show_text=True)
h.show(som)

def som_predict_label(input_vector):
    labels = som.cluster_labels
    predict = som.find_k_nodes([input_vector])[1][0]
    classes = []
    for neighbor in predict:
        classes.append(labels[neighbor])
    return np.bincount(np.array(classes)).argmax()

doc_labels = []
for text in X:
    doc_labels.append(som_predict_label(text))
doc_labels = np.array(doc_labels)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
for label in range(number_of_clusters):
    group = np.where(doc_labels == label)[0]
    text = ' '.join(dataset.loc[group, '$poema'])
    if len(text) > 0:
        wordcloud = WordCloud().generate(text)
        plt.figure(figsize=(16,8))
        plt.title('Nuvem de palavras para o cluster: {}'.format(label))
        plt.imshow(wordcloud)
        plt.axis('off')
    else:
        print('Grupo {} sem nenhum documento associado.'.format(label))
