"""
Funções de gráfico reutilizadas nos notebooks de modelagem - o mesmo motivo do
`src/evaluation/metricas.py`: evitar reescrever o mesmo `matplotlib` em cada
notebook novo.
"""

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

CORES_PADRAO = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]


def plot_comparacao_modelos(tabela, coluna_modelo: str = "Modelo", coluna_valor: str = "Acurácia",
                             titulo: str = "Comparação entre modelos", cores=None):
    """
    Gráfico de barras comparando modelos por uma métrica (default: acurácia).
    Espera uma tabela como a que `comparar_modelos()` devolve.
    """
    cores = cores or CORES_PADRAO[: len(tabela)]
    plt.figure(figsize=(8, 5))
    plt.bar(tabela[coluna_modelo], tabela[coluna_valor], color=cores)
    plt.ylabel(coluna_valor)
    plt.title(titulo)
    plt.ylim(0, 1)
    for i, v in enumerate(tabela[coluna_valor]):
        plt.text(i, v + 0.015, f"{v:.3f}", ha="center")
    plt.show()


def plot_arvore(modelo, feature_names, class_names=("Não", "Sim"), figsize=(22, 10), fontsize=8):
    """
    `plot_tree` com os ajustes de tamanho/fonte que uso pra manter a árvore
    legível, e limpando o prefixo `numericas__`/`categoricas__` que o
    `ColumnTransformer.get_feature_names_out()` devolve (não ajuda em nada na
    leitura do gráfico).
    """
    nomes_limpos = [
        nome.split("__", 1)[1] if "__" in nome else nome
        for nome in feature_names
    ]
    plt.figure(figsize=figsize)
    plot_tree(
        modelo,
        feature_names=nomes_limpos,
        class_names=list(class_names),
        filled=True,
        rounded=True,
        fontsize=fontsize,
    )
    plt.show()


def plot_importancia_features(importancias, top_n: int = 10, titulo: str = "Importância das features"):
    """
    Gráfico de barras horizontais com as `top_n` features mais importantes.
    `importancias` é uma `pd.Series` (índice = nome da feature, valor =
    importância), como as que saem de `feature_importances_` (Random Forest)
    ou dos coeficientes (em módulo) da Regressão Logística.
    """
    top = importancias.sort_values(ascending=False).head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 0.4 * top_n + 1))
    plt.barh(top.index, top.values, color=CORES_PADRAO[0])
    plt.xlabel("Importância")
    plt.title(titulo)
    plt.tight_layout()
    plt.show()
