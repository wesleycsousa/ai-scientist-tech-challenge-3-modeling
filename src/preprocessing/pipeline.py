"""
Pipeline de pré-processamento da modelagem no grão escola - Fase 3.

Junta, num só lugar, a lista final de features que fechei revisando a
tabela da Seção 14.1 do notebook 08 (grupo a grupo, olhando correlação com
o alvo, percentual de preenchimento e a EDA básica de cada coluna) com o
`ColumnTransformer` que já vinha sendo testado na Seção 11 daquele mesmo
notebook - agora com a lista definitiva, não mais a provisória.

A decisão coluna a coluna (mantida x descartada) está registrada em
`reports/referencias/selecao_features_revisado.csv`.

Uso: `python -m src.preprocessing.pipeline` roda um teste rápido de fit no
treino e transform no teste, só pra eu confirmar que a lista final de
features não quebra nada antes de seguir pra modelagem de verdade
(`src/modeling/`).
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.build_base_escola import ler_base_escola_modelo


# Lista final de features numéricas, fechada depois de revisar a tabela da
# Seção 14.1 do notebook 08 grupo a grupo.
FEATURES_NUMERICAS = [
    "quantidade_docente_fundamental_anos_iniciais",
    "profissional_psicologo",
    "quantidade_profissional_pedagogia",
    "profissional_administrativo",
    "quantidade_profissional_coordenador",
    "orgao_associacao_pais_mestres",
    "orgao_gremio_estudantil",
    "orgao_conselho_escolar",
    "etapa_ensino_fundamental_anos_finais",
    "quantidade_equipamento_tv",
    "acesso_internet_computador",
    "tratamento_lixo_inexistente",
    "internet_aprendizagem",
    "quadra_esportes",
    "esgoto_rede_publica",
    "material_pedagogico_desportiva",
    "parque_infantil",
    "sala_professor",
    "internet_alunos",
    "laboratorio_informatica",
    "agua_rede_publica",
    "material_pedagogico_multimidia",
    "refeitorio",
    "biblioteca_sala_leitura",
    "equipamento_impressora_multifuncional",
    "tratamento_lixo_reciclagem",
    "equipamento_tv",
    "equipamento_computador",
    "material_pedagogico_cientifico",
    "biblioteca",
    "laboratorio_ciencias",
    "sala_leitura",
    "banda_larga",
    "etapa_ensino_infantil_creche",
    "etapa_ensino_medio",
    "noturno",
    "redes_sociais",
    "quantidade_desktop_aluno",
]

# Duas colunas ficaram de fora da lista acima por decisão manual de
# multicolinearidade (correlação abaixo do limiar automático de 0,95 usado
# no notebook 08, mas alta o suficiente pra eu preferir manter só uma de
# cada par, revisando a tabela da Seção 14.1 na mão):
# - `acesso_internet_dispositivo_pessoal` saiu, fiquei só com
#   `acesso_internet_computador` (correlação de 0,91 entre as duas);
# - `quadra_esportes_coberta` saiu, fiquei só com `quadra_esportes`
#   (correlação de 0,79 entre as duas).

# Essas colunas numéricas têm um terceiro valor além de 0/1 - o código 9
# ("não informado" no Censo). Decidi tratar esse código como nulo (não como
# um valor numérico de verdade) antes da imputação - do contrário ele
# entraria na mediana/escala como se fosse "mais" que 1, o que não tem
# sentido pra uma flag binária.
COLUNAS_COM_CODIGO_NAO_INFORMADO = [
    "acesso_internet_computador",
    "tratamento_lixo_inexistente",
    "tratamento_lixo_reciclagem",
    "redes_sociais",
]

# Lista final de features categóricas - inclui as que sobreviveram ao funil
# do notebook 08 (dicionário oficial) mais `sigla_uf` e `rede`, que vêm como
# texto do lado do IDEB e não passam pelo dicionário do Censo Escolar.
FEATURES_CATEGORICAS = [
    "tipo_regulamentacao",
    "tipo_responsavel_regulamentacao",
    "tipo_local_funcionamento_predio_escolar",
    "tipo_rede_local",
    "sigla_uf",
    "tipo_localizacao",
    "tipo_aee",
    "tipo_atividade_complementar",
    "rede",
]

FEATURES_FINAIS = FEATURES_NUMERICAS + FEATURES_CATEGORICAS


def carregar_dataset_modelagem(forcar_releitura: bool = False):
    """
    Carrega a base já traduzida (`ler_base_escola_modelo`) e devolve só o
    que entra no modelo: X com a lista final de features, y com
    `alvo_ideb`.

    Filtro obrigatório: mantenho só as escolas com `ideb` preenchido - é o
    alvo do modelo, então uma escola sem `ideb` não serve nem pra treino
    nem pra teste.
    """
    base = ler_base_escola_modelo(forcar_releitura=forcar_releitura)

    mascara_ideb_existe = base["ideb"].notna()
    print(
        f"Escolas com ideb preenchido: {mascara_ideb_existe.sum()} de {len(base)} "
        f"({mascara_ideb_existe.mean() * 100:.1f}%) - só essas entram no dataset de modelagem."
    )

    base_modelagem = base.loc[mascara_ideb_existe].copy()

    faltando = [c for c in FEATURES_FINAIS if c not in base_modelagem.columns]
    if faltando:
        raise KeyError(f"Colunas da lista final que não existem na base: {faltando}")

    # Código 9 ("não informado") vira nulo antes de qualquer imputação -
    # ver COLUNAS_COM_CODIGO_NAO_INFORMADO acima.
    for coluna in COLUNAS_COM_CODIGO_NAO_INFORMADO:
        qtd_nao_informado = (base_modelagem[coluna] == 9).sum()
        if qtd_nao_informado:
            base_modelagem[coluna] = base_modelagem[coluna].replace(9, pd.NA)
        print(f"{coluna}: {qtd_nao_informado} escolas com código 9 (não informado) -> viraram nulo.")

    X = base_modelagem[FEATURES_FINAIS]
    y = base_modelagem["alvo_ideb"].astype(int)
    return X, y


def construir_preprocessador(escalar_numericas: bool = True) -> ColumnTransformer:
    """
    ColumnTransformer com a mesma estratégia de imputação/encoding fechada
    na Seção 11.2 do notebook 08: mediana pras numéricas, categoria
    "desconhecido" + One-Hot pras categóricas.

    `escalar_numericas` controla se entra StandardScaler depois da
    mediana. Modelos baseados em distância/gradiente (Regressão Logística,
    SVM) precisam de escala; árvore e Naive Bayes não - e treinar os dois
    com a mesma escala usada por LR/SVM não teria efeito nenhum na árvore,
    mas também não é o padrão didático que estou seguindo no notebook de
    baseline, daí o parâmetro.
    """
    etapas_numericas = [("imputador", SimpleImputer(strategy="median"))]
    if escalar_numericas:
        etapas_numericas.append(("escala", StandardScaler()))
    transformador_numerico = Pipeline(steps=etapas_numericas)

    transformador_categorico = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="constant", fill_value="desconhecido")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("numericas", transformador_numerico, FEATURES_NUMERICAS),
        ("categoricas", transformador_categorico, FEATURES_CATEGORICAS),
    ])


if __name__ == "__main__":
    X, y = carregar_dataset_modelagem()
    print(f"\nX: {X.shape}, y: {y.shape}")
    print(f"Distribuição do alvo:\n{y.value_counts(normalize=True).round(3)}")

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"\nTreino: {X_treino.shape}, Teste: {X_teste.shape}")

    preprocessador = construir_preprocessador()
    X_treino_transformado = preprocessador.fit_transform(X_treino)
    X_teste_transformado = preprocessador.transform(X_teste)

    print(f"\nX_treino após o ColumnTransformer: {X_treino_transformado.shape}")
    print(f"X_teste após o ColumnTransformer: {X_teste_transformado.shape}")
    print("\nPré-processador rodou sem erro no fit (treino) e no transform (teste).")
