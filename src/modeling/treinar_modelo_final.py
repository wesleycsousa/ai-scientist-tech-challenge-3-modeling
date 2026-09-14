"""
Treina o modelo final do projeto (Árvore de Decisão, Achado 43) de forma
reproduzível, fora de notebook, e salva o pipeline completo em
`models/arvore_decisao_final.joblib`.

É a mesma configuração fechada nos notebooks 10, 11 e 12: pré-processador
oficial (`construir_preprocessador`, sem escala, porque árvore não precisa)
mais `DecisionTreeClassifier(criterion="gini", max_depth=None,
min_samples_leaf=50, random_state=42)`, treinado nos 80% de treino do mesmo
`train_test_split` (`random_state=42`, estratificado) usado em todas as
etapas. Os 20% de validação servem só pra reportar acurácia e AUC no fim,
nunca entram em `.fit()`.

Uso:
    python -m src.modeling.treinar_modelo_final

Depois de treinado, o arquivo salvo é o que `src/modeling/prever.py` carrega
pra gerar previsões em lote sem retreinar nada.
"""

from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.pipeline import carregar_dataset_modelagem, construir_preprocessador

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
CAMINHO_MODELO_FINAL = RAIZ_PROJETO / "models" / "arvore_decisao_final.joblib"

# Hiperparâmetros finais (Achado 43): a configuração mais simples da busca
# leve do notebook 10, que generalizou melhor que a busca pesada do 11.
HIPERPARAMETROS_FINAIS = dict(
    criterion="gini",
    max_depth=None,
    min_samples_leaf=50,
    random_state=42,
)


def construir_modelo_final() -> Pipeline:
    """Pipeline completo: pré-processamento oficial + Árvore de Decisão final."""
    return Pipeline([
        ("prep", construir_preprocessador(escalar_numericas=False)),
        ("clf", DecisionTreeClassifier(**HIPERPARAMETROS_FINAIS)),
    ])


def treinar_e_salvar(caminho_saida: Path = CAMINHO_MODELO_FINAL) -> Pipeline:
    X, y = carregar_dataset_modelagem()
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"Treino: {X_treino.shape}, validação: {X_validacao.shape}")

    modelo = construir_modelo_final()
    modelo.fit(X_treino, y_treino)

    previsao = modelo.predict(X_validacao)
    probabilidade = modelo.predict_proba(X_validacao)[:, 1]
    print(f"Acurácia na validação: {accuracy_score(y_validacao, previsao):.4f} (referência do Achado 43: 0,7315)")
    print(f"AUC na validação: {roc_auc_score(y_validacao, probabilidade):.3f} (referência do notebook 11: 0,791)")

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, caminho_saida)
    tamanho_mb = caminho_saida.stat().st_size / (1024 * 1024)
    print(f"Modelo salvo em {caminho_saida} ({tamanho_mb:.2f} MB)")
    return modelo


if __name__ == "__main__":
    treinar_e_salvar()
