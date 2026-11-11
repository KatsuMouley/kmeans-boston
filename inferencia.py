#Módulo de inferência - receber uma nova residência e determinar o seu cluster
from pickle import load

import pandas as pd

dados_cat_rotulos = ['CHAS', 'RAD']

#Receber os dados de uma nova residência
nova_residencia = {
    'CRIM': float(input('CRIM - taxa de criminalidade per capita: ')),
    'ZN': float(input('ZN - % de terrenos residenciais > 25.000 pés²: ')),
    'INDUS': float(input('INDUS - % de área comercial não varejista: ')),
    'CHAS': int(input('CHAS - margeia o rio Charles (1 = sim, 0 = não): ')),
    'NOX': float(input('NOX - concentração de óxidos nítricos: ')),
    'RM': float(input('RM - nº médio de cômodos: ')),
    'AGE': float(input('AGE - % de imóveis construídos antes de 1940: ')),
    'DIS': float(input('DIS - distância a centros de emprego: ')),
    'RAD': int(input('RAD - índice de acesso a rodovias (1-8 ou 24): ')),
    'TAX': float(input('TAX - imposto predial: ')),
    'PTRATIO': float(input('PTRATIO - razão aluno/professor: ')),
    'B': float(input('B - índice B: ')),
    'LSTAT': float(input('LSTAT - % da população de baixa renda: ')),
}
nova_residencia = pd.DataFrame([nova_residencia])

#Carregar os modelos
modelo_normalizador = load(open('modelo/normalizador.model', 'rb'))
colunas = load(open('modelo/colunas.model', 'rb'))
housing_cluster = load(open('modelo/housing_cluster.model', 'rb'))

#Normalizar os dados da nova residência
nova_num = nova_residencia.drop(columns=dados_cat_rotulos)
nova_num_normalizada = pd.DataFrame(modelo_normalizador.transform(nova_num), columns=nova_num.columns)
nova_cat_normalizada = pd.get_dummies(nova_residencia[dados_cat_rotulos].astype(str), prefix=dados_cat_rotulos, dtype=int)
nova_normalizada = nova_num_normalizada.join(nova_cat_normalizada)
#Garantir as mesmas colunas do treino (categorias ausentes = 0)
nova_normalizada = nova_normalizada.reindex(columns=colunas, fill_value=0)

#Inferir o cluster
cluster = housing_cluster.predict(nova_normalizada)[0]
print(f'\nA nova residência pertence ao cluster {cluster}')

#Desnormalizar o centróide do cluster inferido
centroide = pd.DataFrame([housing_cluster.cluster_centers_[cluster]], columns=colunas)

#Numéricos: inverse_transform
colunas_num = list(modelo_normalizador.feature_names_in_)
centroide_num = pd.DataFrame(modelo_normalizador.inverse_transform(centroide[colunas_num]), columns=colunas_num)

#Categóricos: categoria de maior peso no centróide
#(o centróide é uma média, então os dummies têm valores entre 0 e 1)
centroide_cat = {}
for rotulo in dados_cat_rotulos:
    dummies = [c for c in colunas if c.startswith(rotulo + '_')]
    centroide_cat[rotulo] = int(centroide[dummies].iloc[0].idxmax().split('_')[1])

centroide_desnormalizado = centroide_num.assign(**centroide_cat)[nova_residencia.columns]

print('\nCentróide do cluster (desnormalizado):')
print(centroide_desnormalizado.T.rename(columns={0: 'centróide'}).round(4))
