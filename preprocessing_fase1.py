#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
from collections import Counter


# caminho absoluto da pasta de onde preprocessamento esta sendo executado
ABS_PATH = os.path.dirname(os.path.abspath("__file__"))
DATASETS_DIR = ABS_PATH + "/datasets/{}"


# lista noma dos arquivos de texto
textos = [filename for filename in os.listdir(ABS_PATH + '/textos_originais')
          if filename.endswith('txt')]

# monta dicionario com nome do autor e coletanea de textos
files = {}
for texto in textos:
    with open(ABS_PATH + '/textos_originais/' + texto,  
              encoding='latin1') as f:
        files[texto] = re.sub('\?|!|\.+|,|\"|[|]|(|)|:|-', '',f.read()).lower()

# funcao para remover stopwords
def remove_stopwords(text):
    with open('stopwords.txt','r') as f:
        stopwords = f.read().split('\n')
        for stopword in stopwords:
            text = re.sub(' '+stopword.strip()+' ', ' ', text)
            text = re.sub(' +', ' ', text)
    return text

# para cada texto separa os poemas do mesmo
dataset = []
for texto in textos:
    conteudo = files[texto]
    titulos = re.findall('(?<=\n\n).*\n\n', conteudo)
    titulos = [re.sub('\n+', '', titulo) for titulo in titulos]
    conteudo_temp = re.sub('\n\n.*\n\n', '\n$\n', conteudo).split('\n$\n')[1:]
    autor = re.sub('\.txt|[0-9]+', '', texto)
    for id, text in enumerate(conteudo_temp):
        text = re.sub('\n+',' ', text)
        # stop words
        text = remove_stopwords(text)
        text = re.sub(' +',' ', text)
        dataset.append((titulos[id], text, autor))

# dataset to dataframe
ds = pd.DataFrame(dataset, columns=('titulo','poema','heteronomio'))

# balanceamento das classes
heteronomios = list(ds['heteronomio'].values)
print(Counter(heteronomios))

# salva csv
ds.to_csv(DATASETS_DIR.format('dataset.csv'), index=False)
