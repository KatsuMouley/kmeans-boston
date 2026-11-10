"""
Módulo de treinamento — Clusterização K-Means da base Boston Housing.

Etapas:
1. Carrega a base.
2. Normaliza: dados quantitativos com MinMaxScaler; dados categóricos
   (CHAS e RAD) com One-Hot Encoding.
3. Determina o número ótimo de grupos pelo método do cotovelo
   (maior distância de cada ponto da curva de inércia à reta
   que liga o primeiro e o último ponto).
4. Treina o K-Means com o k ótimo e salva o modelo.
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ARQ_DADOS = "data/BostonHousing.csv"
ARQ_MODELO = "modelo/modelo_kmeans.pkl"
ARQ_COTOVELO = "modelo/cotovelo.png"

# CHAS: variável binária (margeia o rio Charles ou não)
# RAD: índice de acessibilidade a rodovias (classes discretas: 1–8 e 24)
CATEGORICAS = ["chas", "rad"]
# MEDV (valor mediano do imóvel) é o alvo original da base; fica fora da
# clusterização porque uma nova residência não terá esse valor conhecido.
EXCLUIR = ["medv"]


def carregar_dados():
    df = pd.read_csv(ARQ_DADOS)
    df.columns = df.columns.str.lower()
    for c in CATEGORICAS:
        df[c] = df[c].astype(int)
    return df


def criar_normalizador(quantitativas):
    return ColumnTransformer([
        ("quant", MinMaxScaler(), quantitativas),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ])


def k_otimo_cotovelo(inercias, ks):
    """Ponto da curva mais distante da reta entre o primeiro e o último ponto.
    Os eixos são escalados para [0, 1] para que k e inércia tenham o mesmo peso."""
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
    y = (np.array(inercias) - min(inercias)) / (max(inercias) - min(inercias))
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    distancias = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.hypot(x2 - x1, y2 - y1)
    return ks[int(np.argmax(distancias))], list(distancias)


def main():
    df = carregar_dados()
    quantitativas = [c for c in df.columns if c not in CATEGORICAS + EXCLUIR]

    normalizador = criar_normalizador(quantitativas)
    X = normalizador.fit_transform(df[quantitativas + CATEGORICAS])

    # Método do cotovelo
    ks = list(range(1, 11))
    inercias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in ks]
    k, distancias = k_otimo_cotovelo(inercias, ks)

    print("k  | inércia   | distância à reta")
    for kk, i, d in zip(ks, inercias, distancias):
        print(f"{kk:<2} | {i:9.2f} | {d:.2f}")
    print(f"\nNúmero ótimo de grupos: {k}")

    plt.plot(ks, inercias, "o-")
    plt.plot([ks[0], ks[-1]], [inercias[0], inercias[-1]], "--", color="gray")
    plt.axvline(k, color="red", linestyle=":", label=f"k ótimo = {k}")
    plt.xlabel("Número de grupos (k)")
    plt.ylabel("Inércia (soma das distâncias²)")
    plt.title("Método do cotovelo")
    plt.legend()
    plt.savefig(ARQ_COTOVELO, dpi=120, bbox_inches="tight")

    # Treino final e salvamento
    modelo = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    joblib.dump({
        "normalizador": normalizador,
        "kmeans": modelo,
        "quantitativas": quantitativas,
        "categoricas": CATEGORICAS,
    }, ARQ_MODELO)

    df["grupo"] = modelo.labels_
    print("\nResidências por grupo:")
    print(df["grupo"].value_counts().sort_index().to_string())
    print(f"\nModelo salvo em {ARQ_MODELO}")


if __name__ == "__main__":
    main()
