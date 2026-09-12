"""
Funções de avaliação reutilizadas nos notebooks de modelagem (baseline e
otimização) - antes eu repetia o mesmo `accuracy_score`/`classification_report`
em cada notebook, então tirei daqui pra usar em mais de um lugar sem duplicar
código.
"""

from sklearn.metrics import accuracy_score, classification_report


def avaliar_modelo(modelo, X, y, nome: str) -> dict:
    """
    Roda `predict` e imprime a acurácia do modelo num conjunto de dados,
    identificado por `nome`. Devolve um dicionário com o nome, a acurácia e as
    previsões - pra eu poder reaproveitar as previsões depois (matriz de
    confusão, `classification_report`, tabela de comparação) sem rodar
    `predict` de novo.
    """
    previsoes = modelo.predict(X)
    acuracia = accuracy_score(y, previsoes)
    print(f"Acurácia {nome}: {acuracia:.4f}")
    return {"nome": nome, "acuracia": acuracia, "previsoes": previsoes}


def comparar_modelos(resultados: dict, coluna_modelo: str = "Modelo", coluna_valor: str = "Acurácia"):
    """
    Recebe um dicionário {nome_do_modelo: acurácia} e devolve uma tabela
    (DataFrame) ordenada da maior pra menor acurácia, pronta pra exibir ou
    passar pro `plot_comparacao_modelos` (`src/visualization/graficos.py`).
    """
    import pandas as pd

    tabela = pd.DataFrame({
        coluna_modelo: list(resultados.keys()),
        coluna_valor: list(resultados.values()),
    })
    return tabela.sort_values(coluna_valor, ascending=False).reset_index(drop=True)


def relatorio_classificacao_modelos(previsoes: dict, y_verdadeiro, target_names=("Não", "Sim")):
    """
    Imprime o `classification_report` (precisão/recall/f1 por classe) de cada
    modelo em `previsoes` ({nome_do_modelo: array_de_previsoes}), um atrás do
    outro - útil pra comparar mais do que só a acurácia entre os candidatos.
    """
    for nome, previsao in previsoes.items():
        print(f"=== {nome} ===")
        print(classification_report(y_verdadeiro, previsao, target_names=list(target_names)))
