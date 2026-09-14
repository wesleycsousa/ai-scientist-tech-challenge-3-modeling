"""
Previsão em lote com o modelo final salvo (`models/arvore_decisao_final.joblib`).

Carrega a base de escolas já traduzida (`ler_base_escola_modelo`), aplica o
mesmo tratamento de features do treino, gera a probabilidade de cada escola
bater a meta do IDEB e converte isso na faixa de prioridade usada no
notebook 12 (Seção 7): Alta (risco >= 0,90), Média (0,80 a 0,90) e Baixa
(0,50 a 0,80). Escolas com risco abaixo de 0,50 ficam sem prioridade, porque
o modelo já prevê que batem a meta.

Diferente dos notebooks, aqui não filtro pelas escolas que têm `ideb`: o
ponto de um processo em lote é justamente pontuar também as escolas sem
resultado publicado. Quando o `ideb` existe, a coluna `alvo_real` sai
preenchida, pra conferência.

Uso:
    python -m src.modeling.prever
    python -m src.modeling.prever --saida data/processed/previsoes_escolas.csv

O mesmo `prever_escolas()` serve pra uma esteira de produção (job agendado
que reprocessa a base a cada Censo) ou pra um serviço que pontua uma escola
por vez: o pipeline salvo já inclui o pré-processamento, então basta passar
um DataFrame com as mesmas colunas de entrada.
"""

from pathlib import Path
import argparse
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd

from src.preprocessing.build_base_escola import ler_base_escola_modelo
from src.preprocessing.pipeline import COLUNAS_COM_CODIGO_NAO_INFORMADO, FEATURES_FINAIS

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
CAMINHO_MODELO_FINAL = RAIZ_PROJETO / "models" / "arvore_decisao_final.joblib"
CAMINHO_SAIDA_PADRAO = RAIZ_PROJETO / "data" / "processed" / "previsoes_escolas.csv"

COLUNAS_CONTEXTO = ["id_escola", "id_municipio", "sigla_uf", "rede"]


def classificar_prioridade(risco: float):
    """Mesmas faixas do notebook 12 (Seção 7)."""
    if risco >= 0.90:
        return "Alta"
    if risco >= 0.80:
        return "Média"
    if risco >= 0.50:
        return "Baixa"
    return None


def preparar_features(base: pd.DataFrame) -> pd.DataFrame:
    """Mesmo tratamento de entrada do treino: código 9 vira nulo, lista final de colunas."""
    faltando = [c for c in FEATURES_FINAIS if c not in base.columns]
    if faltando:
        raise KeyError(f"Colunas da lista final que não existem na base: {faltando}")
    X = base[FEATURES_FINAIS].copy()
    for coluna in COLUNAS_COM_CODIGO_NAO_INFORMADO:
        X[coluna] = X[coluna].replace(9, pd.NA)
    return X


def carregar_modelo(caminho: Path = CAMINHO_MODELO_FINAL):
    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {caminho}. Rode antes: python -m src.modeling.treinar_modelo_final"
        )
    return joblib.load(caminho)


def prever_escolas(base: pd.DataFrame, modelo=None) -> pd.DataFrame:
    """Devolve um DataFrame com contexto, probabilidade, risco e prioridade por escola."""
    modelo = modelo or carregar_modelo()
    X = preparar_features(base)
    probabilidade_meta = modelo.predict_proba(X)[:, 1]

    resultado = base[[c for c in COLUNAS_CONTEXTO if c in base.columns]].copy()
    resultado["prob_bater_meta"] = probabilidade_meta.round(4)
    resultado["risco_previsto"] = (1 - probabilidade_meta).round(4)
    resultado["prioridade"] = resultado["risco_previsto"].apply(classificar_prioridade)
    if "alvo_ideb" in base.columns:
        resultado["alvo_real"] = base["alvo_ideb"]
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Previsão em lote do risco de não bater a meta do IDEB, por escola.")
    parser.add_argument("--saida", type=Path, default=CAMINHO_SAIDA_PADRAO, help="CSV de saída.")
    parser.add_argument("--modelo", type=Path, default=CAMINHO_MODELO_FINAL, help="Arquivo .joblib do modelo.")
    args = parser.parse_args()

    base = ler_base_escola_modelo()
    modelo = carregar_modelo(args.modelo)
    resultado = prever_escolas(base, modelo)

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(args.saida, index=False)

    em_risco = resultado["prioridade"].notna()
    print(f"Escolas pontuadas: {len(resultado):,}")
    print(f"Em risco (risco_previsto >= 0,50): {em_risco.sum():,} ({em_risco.mean() * 100:.1f}%)")
    print(resultado.loc[em_risco, "prioridade"].value_counts().reindex(["Alta", "Média", "Baixa"]).to_string())
    print(f"Arquivo salvo em {args.saida}")


if __name__ == "__main__":
    main()
