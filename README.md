# Clusterização KMeans — Boston Housing

Avaliação do 2º bimestre: agrupamento de residências da base Boston Housing (`HousingData.csv`) com KMeans.

## Estrutura

```
data/HousingData.csv   # base da aula
treinamento.py         # módulo de treinamento
inferencia.py          # módulo de inferência
modelo/                # normalizador, colunas, modelo KMeans e gráfico do cotovelo
requirements.txt
```

## Como executar

```bash
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
python treinamento.py   # treina e salva os modelos em modelo/
python inferencia.py    # classifica uma nova residência
```

## Decisões

- **Dados faltantes (NA)**: preenchidos com a mediana (quantitativos) e a moda (categóricos).
- **Normalização**:
  - Quantitativos (`CRIM, ZN, INDUS, NOX, RM, AGE, DIS, TAX, PTRATIO, B, LSTAT`): `MinMaxScaler`.
  - Categóricos (`CHAS`, que é binário, e `RAD`, um índice discreto de acesso a rodovias): `pd.get_dummies`.
- **MEDV** (preço) é o alvo original da base. Ficou fora da clusterização, porque uma nova residência não teria esse valor.
- **Balanceamento**: não aplicado, conforme o enunciado.
- **Número de clusters**: método do cotovelo sobre as distorções (média das distâncias euclidianas ao centróide mais próximo), testando k de 1 a 20. O k ótimo é o ponto de maior distância à reta entre o primeiro e o último ponto. Resultado: **k = 4** (`modelo/cotovelo.png`).
- **Desnormalização do centróide**:
  - Quantitativos: `inverse_transform`.
  - Categóricos: a categoria de maior peso entre os dummies, já que o centróide é uma média com valores entre 0 e 1.
