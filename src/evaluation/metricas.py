"""
Funções de avaliação reutilizadas nos notebooks de modelagem (baseline e
otimização) - antes eu repetia o mesmo `accuracy_score`/`classification_report`
em cada notebook, então tirei daqui pra usar em mais de um lugar sem duplicar
código.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_val_predict


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


def obter_previsoes_cv(modelos: dict, X_treino, y_treino, cv_por_modelo: dict,
                        tamanho_amostra: dict = None, random_state: int = 42):
    """
    Previsões via validação cruzada dentro do treino (`cross_val_predict`),
    usando o mesmo `cv` que cada busca de hiperparâmetro já usou (`cv_padrao`
    ou `cv_svm`). Não toca na base de validação (o cofre fechado) - é a
    estimativa de "treino+teste" (o teste de cada fold da validação cruzada),
    não a validação final.

    `tamanho_amostra` (ex.: `{"SVM": 0.4}`) deixa eu passar uma amostra
    estratificada só do treino pros modelos mais caros de recalcular - o
    hiperparâmetro já está fechado nessa etapa, então não é uma busca nova,
    só recalculo previsões pra ter as métricas completas (F1, precisão,
    recall) sem esperar o tempo cheio de novo.
    """
    tamanho_amostra = tamanho_amostra or {}
    previsoes, y_usado = {}, {}
    for nome, modelo in modelos.items():
        fracao = tamanho_amostra.get(nome)
        if fracao and fracao < 1.0:
            X_amostra = X_treino.sample(frac=fracao, random_state=random_state)
            y_amostra = y_treino.loc[X_amostra.index]
        else:
            X_amostra, y_amostra = X_treino, y_treino
        previsoes[nome] = cross_val_predict(modelo, X_amostra, y_amostra, cv=cv_por_modelo[nome])
        y_usado[nome] = y_amostra
    return previsoes, y_usado


def comparar_metricas_completas(previsoes_por_base: dict, nomes_classe=("Não", "Sim")):
    """
    `previsoes_por_base` no formato `{"Nome da base": {"Modelo": (y_verdadeiro,
    y_previsto)}}` - por exemplo `{"Treino (CV)": {...}, "Validação": {...}}`.

    Monta uma tabela longa (Modelo, Base, Classe, Acurácia, Precisão, Recall,
    F1-score) pra comparar lado a lado a performance de cada modelo nas
    diferentes bases - a ideia é enxergar se algum modelo generaliza mal
    (métricas bem piores na validação do que no treino/CV) antes de escolher
    o modelo final.

    Uso `precision_recall_fscore_support` com `labels=[0, 1]` em vez de
    `precision_score`/`recall_score`/`f1_score` direto: essas três funções
    usam por padrão `average="binary"`, que só calcula a métrica pra classe
    1 ("Sim") e ignora a classe 0 ("Não") - eu queria justamente enxergar as
    duas classes separadas, igual o `classification_report` mostra, então
    cada linha da tabela agora é um par (Modelo, Base, Classe).
    """
    linhas = []
    for base, modelos in previsoes_por_base.items():
        for nome, (y_verdadeiro, y_previsto) in modelos.items():
            acuracia = accuracy_score(y_verdadeiro, y_previsto)
            precisoes, recalls, f1s, _ = precision_recall_fscore_support(
                y_verdadeiro, y_previsto, labels=[0, 1], zero_division=0
            )
            for classe_idx, nome_classe in enumerate(nomes_classe):
                linhas.append({
                    "Modelo": nome,
                    "Base": base,
                    "Classe": nome_classe,
                    "Acurácia": acuracia,
                    "Precisão": precisoes[classe_idx],
                    "Recall": recalls[classe_idx],
                    "F1-score": f1s[classe_idx],
                })
    tabela = pd.DataFrame(linhas)
    return tabela.sort_values(["Modelo", "Base", "Classe"]).reset_index(drop=True)


def relatorio_classificacao_modelos(previsoes: dict, y_verdadeiro, target_names=("Não", "Sim")):
    """
    Imprime o `classification_report` (precisão/recall/f1 por classe) de cada
    modelo em `previsoes` ({nome_do_modelo: array_de_previsoes}), um atrás do
    outro - útil pra comparar mais do que só a acurácia entre os candidatos.
    """
    for nome, previsao in previsoes.items():
        print(f"=== {nome} ===")
        print(classification_report(y_verdadeiro, previsao, target_names=list(target_names)))
