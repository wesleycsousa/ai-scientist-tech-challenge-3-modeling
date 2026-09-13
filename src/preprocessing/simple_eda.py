"""
Função de EDA rápida pra bater o olho em qualquer dataset e pegar cedo um
problema de categorização/tradução - motivada por um bug real que só
percebi olhando o output do notebook 08 na tela (colunas que deveriam estar
traduzidas em texto e continuavam aparecendo como código numérico, ou nem
tinham sido tocadas pela tradução - ver Achado 31 em reports/decisoes.md).

Em vez de reescrever esse tipo de checagem toda vez que eu carrego uma base
nova, centralizo aqui uma função só (`simple_eda`) que qualquer notebook
pode chamar.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def simple_eda(
    df: pd.DataFrame,
    limiar_cardinalidade_suspeita: int = 15,
    top_n_categorias: int = 3,
    colunas: Optional[Sequence[str]] = None,
) -> dict:
    """
    Bate o olho rápido em um DataFrame: separa colunas numéricas de
    categóricas (por `dtype` - é o critério que voltou a funcionar bem
    depois que a tradução de código->texto passa a rodar antes de qualquer
    EDA, ver Achado 30/31 em reports/decisoes.md) e traz estatística básica
    de cada grupo.

    Parâmetros
    ----------
    df: DataFrame a inspecionar.
    limiar_cardinalidade_suspeita: colunas NUMÉRICAS com essa quantidade de
        valores distintos (ou menos) são marcadas como "possível código
        categórico não traduzido" - não são movidas pro grupo categórico
        automaticamente (Achado 24: baixa cardinalidade sozinha não prova
        que é categórica, várias contagens legítimas têm poucos valores
        distintos), só sinalizadas pra eu conferir.
    top_n_categorias: quantos valores mais frequentes trazer por coluna
        categórica (padrão 3, como pedido).
    colunas: se passado, limita a EDA só a essas colunas do DataFrame; por
        padrão, roda em todas.

    Retorno
    -------
    Um dicionário com duas chaves, cada uma um DataFrame pronto pra
    inspecionar (`.head()`/`print`):

    - "numericas": índice = nome da coluna; colunas = contagem
      (não nulo), média, mediana, desvio padrão, mínimo, percentil 25,
      percentil 75, máximo, % de nulo, cardinalidade (nº de valores
      distintos) e uma flag `possivel_categoria_nao_traduzida`.
    - "categoricas": índice = nome da coluna; colunas = cardinalidade,
      % de nulo, e (valor, contagem) dos `top_n_categorias` valores mais
      frequentes, em colunas separadas (`top_1_valor`, `top_1_contagem`,
      `top_2_valor`, `top_2_contagem`, ...).

    Uso típico pra pegar um problema de tradução/categorização, tipo o que
    encontrei no notebook 08 (Achado 31): rodar `simple_eda(base)` logo
    depois de carregar a base e olhar se alguma coluna que eu esperava ver
    como categórica (texto) apareceu em "numericas" com
    `possivel_categoria_nao_traduzida=True`, ou se alguma "categoricas"
    mostra valores tipo "codigo_nao_documentado_..." nos top 3 - os dois
    são sinal de que a tradução não rodou (ou rodou errado) pra essa
    coluna.
    """
    colunas_alvo = list(colunas) if colunas is not None else list(df.columns)

    linhas_numericas = []
    linhas_categoricas = []

    for coluna in colunas_alvo:
        serie = df[coluna]
        pct_nulo = round(serie.isna().mean() * 100, 2)
        cardinalidade = serie.nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(serie):
            descricao = serie.describe(percentiles=[0.25, 0.5, 0.75])
            linhas_numericas.append({
                "coluna": coluna,
                "contagem_nao_nulo": int(descricao.get("count", 0)),
                "media": descricao.get("mean", np.nan),
                "mediana": descricao.get("50%", np.nan),
                "desvio_padrao": descricao.get("std", np.nan),
                "minimo": descricao.get("min", np.nan),
                "percentil_25": descricao.get("25%", np.nan),
                "percentil_75": descricao.get("75%", np.nan),
                "maximo": descricao.get("max", np.nan),
                "pct_nulo": pct_nulo,
                "cardinalidade": cardinalidade,
                "possivel_categoria_nao_traduzida": bool(cardinalidade <= limiar_cardinalidade_suspeita),
            })
        else:
            contagens = serie.value_counts(dropna=True).head(top_n_categorias)
            linha = {
                "coluna": coluna,
                "cardinalidade": cardinalidade,
                "pct_nulo": pct_nulo,
            }
            for i in range(top_n_categorias):
                if i < len(contagens):
                    linha[f"top_{i + 1}_valor"] = contagens.index[i]
                    linha[f"top_{i + 1}_contagem"] = int(contagens.iloc[i])
                else:
                    linha[f"top_{i + 1}_valor"] = None
                    linha[f"top_{i + 1}_contagem"] = None
            linhas_categoricas.append(linha)

    df_numericas = pd.DataFrame(linhas_numericas).set_index("coluna") if linhas_numericas else pd.DataFrame()
    df_categoricas = pd.DataFrame(linhas_categoricas).set_index("coluna") if linhas_categoricas else pd.DataFrame()

    n_suspeitas = int(df_numericas["possivel_categoria_nao_traduzida"].sum()) if len(df_numericas) else 0
    print(f"simple_eda: {len(colunas_alvo)} colunas ({len(df_numericas)} numéricas, {len(df_categoricas)} categóricas)")
    if n_suspeitas:
        print(f"⚠️  {n_suspeitas} coluna(s) numérica(s) com baixa cardinalidade "
              f"(<= {limiar_cardinalidade_suspeita}) - possível código categórico não traduzido, "
              f"conferir antes de assumir que é numérica de verdade.")

    return {"numericas": df_numericas, "categoricas": df_categoricas}


if __name__ == "__main__":
    # teste rápido com dado sintético, só pra conferir que a função roda
    exemplo = pd.DataFrame({
        "idade": [10, 12, 11, 10, np.nan, 13, 12],
        "rede": ["municipal", "estadual", "municipal", "municipal", "federal", "estadual", "municipal"],
        "tipo_localizacao": [1, 2, 1, 1, 1, 2, 1],  # exemplo de código não traduzido, cardinalidade baixa
    })
    resultado = simple_eda(exemplo)
    print("\n--- numericas ---")
    print(resultado["numericas"])
    print("\n--- categoricas ---")
    print(resultado["categoricas"])
