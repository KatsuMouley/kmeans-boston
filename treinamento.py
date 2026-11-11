#Módulo de treinamento - Clusterização da base Boston Housing com KMeans
import math
from pickle import dump

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.cluster import KMeans

dados = pd.read_csv('data/HousingData.csv', sep=',')

#Tratar dados faltantes (NA)
#Quantitativos: mediana | Categóricos: moda
dados_cat_rotulos = ['CHAS', 'RAD']  #CHAS: margeia o rio (0/1) | RAD: índice de acesso a rodovias
for col in dados.columns:
    if col in dados_cat_rotulos:
        dados[col] = dados[col].fillna(dados[col].mode()[0])
    else:
        dados[col] = dados[col].fillna(dados[col].median())
for col in dados_cat_rotulos:
    dados[col] = dados[col].astype(int)

#MEDV (preço) é o alvo original da base: fica fora da clusterização,
#pois uma nova residência não terá esse valor
dados = dados.drop(columns=['MEDV'])

#Separar dados numéricos e categóricos
dados_num = dados.drop(columns=dados_cat_rotulos)
dados_cat = dados[dados_cat_rotulos]

#NORMALIZAR DADOS NUMÉRICOS
normalizador = preprocessing.MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_num)
dump(modelo_normalizador, open('modelo/normalizador.model', 'wb'))
dados_num_normalizados = pd.DataFrame(modelo_normalizador.transform(dados_num), columns=dados_num.columns)

#NORMALIZAR DADOS CATEGÓRICOS
dados_cat_normalizados = pd.get_dummies(dados_cat.astype(str), prefix=dados_cat_rotulos, dtype=int)

#Juntar os dados normalizados
dados_normalizados = dados_num_normalizados.join(dados_cat_normalizados)
#Salvar a ordem das colunas para a inferência
dump(list(dados_normalizados.columns), open('modelo/colunas.model', 'wb'))

#Descobrir o número ideal de clusters - distorções (distâncias)
distorcoes = []
K = range(1, 21)
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42, n_init=10).fit(dados_normalizados)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_normalizados, cluster_model.cluster_centers_, 'euclidean'),
                axis=1) / dados_normalizados.shape[0]
        )
    )

#MÉTODO DO COTOVELO (ELBOW)
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
    distancias.append(numerador / denominador)

n_clusters_otimo = K[distancias.index(max(distancias))]
print(f'Número ótimo de clusters: {n_clusters_otimo}')

#Gráfico do cotovelo
fig, ax = plt.subplots()
ax.plot(K, distorcoes, 'o-')
ax.axvline(n_clusters_otimo, color='red', linestyle=':', label=f'k ótimo = {n_clusters_otimo}')
ax.set(xlabel='n Clusters', ylabel='Distorções', title='Método do cotovelo')
ax.legend()
ax.grid()
plt.savefig('modelo/cotovelo.png', dpi=120, bbox_inches='tight')

#Treinamento definitivo
housing_cluster = KMeans(n_clusters=n_clusters_otimo, random_state=42, n_init=10).fit(dados_normalizados)

#Salvar o modelo
dump(housing_cluster, open('modelo/housing_cluster.model', 'wb'))
print('Residências por cluster:', np.bincount(housing_cluster.labels_))
