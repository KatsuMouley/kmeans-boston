"""
Módulo de inferência — classifica uma nova residência em um grupo.

Etapas:
1. Recebe os dados de uma nova residência (via terminal).
2. Normaliza com o mesmo normalizador usado no treinamento.
3. Infere o grupo com o modelo K-Means salvo.
4. Exibe o centroide do grupo inferido desnormalizado.
"""
import joblib
import numpy as np
import pandas as pd

ARQ_MODELO = "modelo/modelo_kmeans.pkl"

DESCRICOES = {
    "crim": "Taxa de criminalidade per capita",
    "zn": "% de terrenos residenciais > 25.000 pés²",
    "indus": "% de área comercial não varejista",
    "chas": "Margeia o rio Charles (1 = sim, 0 = não)",
    "nox": "Concentração de óxidos nítricos (ppm)",
    "rm": "Nº médio de cômodos",
    "age": "% de imóveis construídos antes de 1940",
    "dis": "Distância ponderada a centros de emprego",
    "rad": "Índice de acesso a rodovias (1-8 ou 24)",
    "tax": "Imposto predial por US$ 10.000",
    "ptratio": "Razão aluno/professor",
    "b": "Índice B (proporção populacional)",
    "lstat": "% da população de baixa renda",
}


def receber_dados(colunas):
    print("Informe os dados da nova residência:")
    dados = {}
    for c in colunas:
        while True:
            try:
                dados[c] = float(input(f"  {c} - {DESCRICOES[c]}: ").replace(",", "."))
                break
            except ValueError:
                print("    Valor inválido, digite um número.")
    return pd.DataFrame([dados])


def desnormalizar_centroide(centroide, normalizador, quantitativas, categoricas):
    n = len(quantitativas)
    # Parte quantitativa: inverso do MinMaxScaler
    quant = normalizador.named_transformers_["quant"].inverse_transform(centroide[:n].reshape(1, -1))[0]
    resultado = dict(zip(quantitativas, quant))
    # Parte categórica: categoria de maior peso no one-hot de cada variável
    encoder = normalizador.named_transformers_["cat"]
    pos = n
    for col, cats in zip(categoricas, encoder.categories_):
        bloco = centroide[pos:pos + len(cats)]
        resultado[col] = cats[int(np.argmax(bloco))]
        pos += len(cats)
    return resultado


def main():
    pacote = joblib.load(ARQ_MODELO)
    normalizador = pacote["normalizador"]
    kmeans = pacote["kmeans"]
    quant, cat = pacote["quantitativas"], pacote["categoricas"]
    colunas = quant + cat

    nova = receber_dados([c for c in DESCRICOES if c in colunas])[colunas]
    X = normalizador.transform(nova)
    grupo = int(kmeans.predict(X)[0])

    centroide = desnormalizar_centroide(kmeans.cluster_centers_[grupo], normalizador, quant, cat)

    print(f"\nA nova residência pertence ao GRUPO {grupo}.")
    print("\nCentroide do grupo (valores desnormalizados):")
    print(f"{'Atributo':<9} {'Nova residência':>16} {'Centroide':>12}")
    for c in DESCRICOES:
        print(f"{c:<9} {nova[c].iloc[0]:>16.4f} {float(centroide[c]):>12.4f}")


if __name__ == "__main__":
    main()
