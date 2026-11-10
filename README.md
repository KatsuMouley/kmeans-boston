# Clusterização K-Means — Boston Housing

Avaliação do 2º bimestre: agrupamento de residências da base Boston Housing com K-Means.

## Estrutura

```
data/BostonHousing.csv   # base de dados
treinamento.py           # módulo de treinamento
inferencia.py            # módulo de inferência
modelo/                  # modelo salvo + gráfico do cotovelo
requirements.txt
```

## Como executar

```bash
pip install -r requirements.txt
python treinamento.py   # treina e salva o modelo em modelo/
python inferencia.py    # classifica uma nova residência
```

## Decisões

**Normalização**
- Quantitativas (`crim, zn, indus, nox, rm, age, dis, tax, ptratio, b, lstat`): `MinMaxScaler` (0 a 1).
- Categóricas: `chas` (binária) e `rad` (índice discreto de acesso a rodovias) com `OneHotEncoder`, para que o K-Means não trate as classes como distâncias numéricas.
- `medv` (preço) é o alvo original da base e ficou fora da clusterização, pois uma nova residência não teria esse valor.

**Número de grupos**: método do cotovelo sobre a inércia (soma das distâncias² aos centroides) para k = 1..10. O k ótimo é o ponto da curva com maior distância à reta que liga o primeiro e o último ponto. O gráfico fica em `modelo/cotovelo.png`.

**Balanceamento**: não aplicado (conforme enunciado).

**Desnormalização do centroide**: parte quantitativa via `inverse_transform` do `MinMaxScaler`; parte categórica pela categoria de maior peso no one-hot.
