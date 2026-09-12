"""
Construção da base "silver" de escola, já filtrada e com o alvo definido.

Este script formaliza, em código reutilizável, o que fiz de forma
exploratória na Seção 1 do `notebooks/eda/07_eda_escola_silver.ipynb` —
agora que o ano de alinhamento (2025) deixou de ser provisório e virou
decisão final (`reports/decisoes.md`, Achado 17), faz sentido esse
pipeline sair do notebook e virar uma função testável e reaproveitável,
em vez de ficar reescrevendo a mesma lógica em célula de notebook toda
vez que eu precisar da base.

A base que este módulo monta é o resultado de toda a investigação dos
notebooks `03` a `07` (ver `reports/decisoes.md`, Achados 10 a 19):

- Grão: escola (`id_escola`), não aluno.
- Alvo: `alvo_ideb = 1 se ideb >= 6.0, senão 0` (corte = meta nacional do
  MEC/INEP, não um valor arbitrário).
- Filtros aplicados (nessa ordem, todos decisões já fechadas):
    1. Só escolas que aparecem tanto no `br_inep_ideb/escola` quanto na
       `escola_completo` (crosswalk validado, 95,89% de interseção).
    2. Só rede pública (municipal + estadual + federal) - a privada tem
       preenchimento muito pior de resultado e é <0,5% da base.
    3. Só a etapa "anos iniciais (1-5)" - a mais próxima da alfabetização
       dentro do que o IDEB oferece.
    4. Ano do IDEB = 2025, pareado com o Censo Escolar de 2024 (infra
       de escola muda pouco de um ano pro outro - ver Achado 17).

As colunas que compõem o próprio IDEB (`taxa_aprovacao`,
`indicador_rendimento`, as notas do SAEB, `ideb`, `projecao`) continuam
na base retornada aqui, porque fazem parte do registro/auditoria da
variável-alvo - mas NÃO podem entrar como feature de nenhum modelo,
exatamente pelo motivo já verificado no Achado 15 (elas compõem o
próprio alvo). Uso `COLUNAS_ALVO_EXCLUIR_DE_FEATURES` no fim deste
módulo pra deixar isso explícito e fácil de importar em outro lugar.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

# Rodar este arquivo direto (botão "Run" do VS Code, ou `python
# build_base_escola.py`) executa o script com `src/preprocessing/` como
# working directory do import, não a raiz do repositório - o Python não
# acha o pacote `src` nesse caso (`ModuleNotFoundError: No module named
# 'src'`). Rodar com `python -m src.preprocessing.build_base_escola` a
# partir da raiz do repo evita isso, mas prefiro deixar o arquivo robusto
# aos dois jeitos de rodar - adiciono a raiz do repo no sys.path só quando
# este arquivo é executado diretamente (`__name__ == "__main__"`); quando é
# importado por um notebook ou por outro módulo, quem importa já cuida do
# próprio sys.path, então não mexo em nada aqui.
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.load_data import BUCKET, _ler_parquet_do_prefixo, _salvar_parquet_no_prefixo

# Decisão final, documentada em reports/decisoes.md (Achado 17): o Censo
# Escolar (escola_completo) só existe pra ano=2024, e o IDEB só é
# calculado em anos ímpares. Uso o ciclo de 2025 do IDEB pareado com a
# infraestrutura de 2024 - infraestrutura escolar muda pouco de um ano
# pro outro, então essa combinação é uma aproximação razoável.
ANO_ESCOLHIDO = 2025

PREFIXO_BRONZE_IDEB_ESCOLA = "bronze/br_inep_ideb/escola/"
PREFIXO_BRONZE_ESCOLA_COMPLETO = "bronze/br_inep_censo_escolar/escola_completo/"
# Ingerida em 2026-09-10, resolvendo a pendência de tradução de código
# registrada no Achado 24 de reports/decisoes.md (na época eu só tinha
# uma query especulativa via `basedosdados`, sem confirmação de que
# funcionava - agora é uma tabela real já no nosso bronze).
PREFIXO_BRONZE_DICIONARIO_ESCOLA = "bronze/br_inep_censo_escolar/dicionario/"

# Camada nova (decisão registrada em reports/decisoes.md): a junção com o
# dicionário de tradução deixa de acontecer dentro do notebook e passa a
# fazer parte do pré-processamento. O resultado (base pronta pra modelagem,
# já com as categorias traduzidas) é publicado aqui, num prefixo próprio -
# diferente dos prefixos "bronze/" acima (que só leio, veio de ingestão
# externa), este eu mesmo escrevo, como saída deste módulo.
PREFIXO_SILVER_MODELO_BASE_ESCOLA = "silver_modelo/br_inep_censo_escolar/base_escola_modelo/"

# Mesmos caches locais já usados nos notebooks 06/07 - se eu já rodei
# aqueles notebooks antes, esses arquivos já existem e a leitura é
# instantânea (sem baixar do S3 de novo).
CACHE_IDEB_ESCOLA = Path(__file__).resolve().parents[2] / "data" / "processed" / "ideb_escola.parquet"
CACHE_ESCOLA_COMPLETO = Path(__file__).resolve().parents[2] / "data" / "processed" / "escola_completo.parquet"
CACHE_BASE_ESCOLA_MODELAGEM = Path(__file__).resolve().parents[2] / "data" / "processed" / "base_escola_modelagem.parquet"
CACHE_DICIONARIO_ESCOLA = Path(__file__).resolve().parents[2] / "data" / "processed" / "dicionario_escola.parquet"
CACHE_BASE_ESCOLA_MODELO = Path(__file__).resolve().parents[2] / "data" / "processed" / "base_escola_modelo.parquet"

# Valor usado quando um código aparece no dado real mas não tem tradução
# no dicionário. Não deveria ser o caso normal - já testei a cobertura
# do dicionário contra `escola_completo.parquet` de verdade antes de
# escrever esta função (ver reports/decisoes.md) e a única coluna com
# gap foi `tipo_localizacao_diferenciada` (códigos 0 e 8 sem tradução na
# amostra que testei) - então trato isso como esperado, não como bug,
# mas deixo visível em vez de mascarar.
VALOR_CODIGO_NAO_DOCUMENTADO = "codigo_nao_documentado"

# Essas colunas compõem o próprio IDEB (Achado 15 em reports/decisoes.md)
# - ficam na base retornada aqui como registro/auditoria da variável-alvo,
# mas nunca podem entrar como feature de um modelo (seria vazar o alvo
# pra dentro das próprias entradas do modelo).
COLUNAS_ALVO_EXCLUIR_DE_FEATURES = [
    "taxa_aprovacao",
    "indicador_rendimento",
    "nota_saeb_matematica",
    "nota_saeb_lingua_portuguesa",
    "nota_saeb_media_padronizada",
    "ideb",
    "projecao",
]


def ler_ideb_escola(forcar_releitura: bool = False) -> pd.DataFrame:
    """
    Lê o `br_inep_ideb/escola` (grão: id_escola, ano, rede, ensino,
    anos_escolares), com cache local em parquet.
    """
    if CACHE_IDEB_ESCOLA.exists() and not forcar_releitura:
        print(f"Lendo ideb_escola do cache local: {CACHE_IDEB_ESCOLA}")
        return pd.read_parquet(CACHE_IDEB_ESCOLA)

    print(f"Cache não encontrado, lendo do S3: s3://{BUCKET}/{PREFIXO_BRONZE_IDEB_ESCOLA}")
    df = _ler_parquet_do_prefixo(BUCKET, PREFIXO_BRONZE_IDEB_ESCOLA)
    CACHE_IDEB_ESCOLA.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_IDEB_ESCOLA, index=False)
    print(f"Cache salvo em: {CACHE_IDEB_ESCOLA}")
    return df


def ler_escola_completo(forcar_releitura: bool = False) -> pd.DataFrame:
    """
    Lê a `escola_completo` (Censo Escolar, 455 colunas, grão: id_escola,
    ano=2024 fixo), com cache local em parquet.
    """
    if CACHE_ESCOLA_COMPLETO.exists() and not forcar_releitura:
        print(f"Lendo escola_completo do cache local: {CACHE_ESCOLA_COMPLETO}")
        return pd.read_parquet(CACHE_ESCOLA_COMPLETO)

    print(f"Cache não encontrado, lendo do S3: s3://{BUCKET}/{PREFIXO_BRONZE_ESCOLA_COMPLETO}")
    df = _ler_parquet_do_prefixo(BUCKET, PREFIXO_BRONZE_ESCOLA_COMPLETO)
    CACHE_ESCOLA_COMPLETO.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_ESCOLA_COMPLETO, index=False)
    print(f"Cache salvo em: {CACHE_ESCOLA_COMPLETO}")
    return df


def ler_dicionario_escola(forcar_releitura: bool = False) -> pd.DataFrame:
    """
    Lê o dicionário de tradução código->texto das colunas categóricas do
    Censo Escolar (colunas: dataset_id, id_tabela, nome_coluna, chave,
    valor), com cache local em parquet - mesmo padrão das outras leituras
    deste módulo.
    """
    if CACHE_DICIONARIO_ESCOLA.exists() and not forcar_releitura:
        print(f"Lendo dicionario_escola do cache local: {CACHE_DICIONARIO_ESCOLA}")
        df = pd.read_parquet(CACHE_DICIONARIO_ESCOLA)
    else:
        print(f"Cache não encontrado, lendo do S3: s3://{BUCKET}/{PREFIXO_BRONZE_DICIONARIO_ESCOLA}")
        df = _ler_parquet_do_prefixo(BUCKET, PREFIXO_BRONZE_DICIONARIO_ESCOLA)
        CACHE_DICIONARIO_ESCOLA.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE_DICIONARIO_ESCOLA, index=False)
        print(f"Cache salvo em: {CACHE_DICIONARIO_ESCOLA}")

    # Bug real que só apareceu inspecionando o dicionário de verdade
    # (não aparecia na amostra que eu tinha usado pra testar antes): várias
    # linhas com id_tabela="escola" têm espaço sobrando no fim de
    # `nome_coluna` (ex.: "tipo_situacao_funcionamento " em vez de
    # "tipo_situacao_funcionamento") - isso nunca batia com o nome real da
    # coluna na base, e a coluna ficava sem tradução nenhuma, sem erro
    # nenhum (silenciosamente errado - o pior tipo de bug). Normalizo aqui,
    # uma vez só, pra todo mundo que usar esse dicionário já receber os
    # nomes limpos.
    df["nome_coluna"] = df["nome_coluna"].str.strip()
    df["id_tabela"] = df["id_tabela"].str.strip()
    return df


def _normalizar_chave_codigo(valor):
    """
    Normaliza uma chave de código (tanto do dicionário quanto do dado real)
    pra um formato comum de comparação: int quando dá pra converter (cobre
    1, 1.0, "1", "1.0" - todos viram o int 1), senão string.

    Bug real encontrado inspecionando o dicionário de verdade: a coluna
    `chave` do dicionário ingerido vem como *string* (`"1"`, `"2"`...),
    enquanto o código anterior comparava com um int puro calculado a partir
    do valor da base - a comparação nunca batia (tipos diferentes), e toda
    tradução caía no fallback "codigo_nao_documentado_<código>", mesmo pros
    códigos que o dicionário documenta certinho. Comparar as duas pontas já
    normalizadas pelo mesmo critério resolve isso.
    """
    if pd.isna(valor):
        return None
    try:
        return int(float(valor))
    except (ValueError, TypeError):
        return str(valor).strip()


def colunas_categoricas_com_dicionario(dicionario: pd.DataFrame, id_tabela: str = "escola") -> List[str]:
    """
    Lista definitiva de colunas categóricas da tabela `escola`: em vez de
    adivinhar por prefixo de nome (heurística que usei antes de ter o
    dicionário oficial - ver Achado 24 em reports/decisoes.md), uso a
    própria cobertura do dicionário. Qualquer coluna listada aqui tem
    código->texto documentado pelo INEP/Base dos Dados - não é palpite.
    """
    subset = dicionario[dicionario["id_tabela"] == id_tabela]
    return sorted(subset["nome_coluna"].unique().tolist())


def traduzir_categoricas_escola(
    df: pd.DataFrame,
    dicionario: pd.DataFrame,
    id_tabela: str = "escola",
    colunas: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Substitui os códigos numéricos das colunas categóricas pelo texto
    oficial do dicionário - deixa o resultado pronto para o
    `OneHotEncoder` com nomes de feature legíveis (Seção 11.2 do
    notebook 08), em vez de categorias tipo "tipo_regulamentacao_2.0".

    As colunas de código em `escola_completo` vêm como *string*
    representando float (ex.: "1.0", "2.0"), não como int puro -
    confirmei isso rodando um teste real contra
    `data/processed/escola_completo.parquet` antes de escrever esta
    função (não é um palpite de dtype). A coluna `chave` do dicionário, por
    sua vez, vem como *string* também (`"1"`, `"2"`...) - confirmado
    inspecionando o parquet de verdade. Por isso os dois lados passam pelo
    mesmo normalizador (`_normalizar_chave_codigo`) antes de comparar, em
    vez de cada lado converter do seu jeito (bug real que já apareceu
    aqui - ver Achado 31 em reports/decisoes.md).

    Nulo vira a string "desconhecido" (não uso `NaN` porque depois do
    `OneHotEncoder` eu quero uma categoria própria e legível pra "não
    respondeu", igual já documentei na Seção 11.2). Um código que existe
    no dado mas não tem tradução no dicionário vira
    "codigo_nao_documentado_<código>", em vez de quebrar ou virar nulo
    silenciosamente - já sei que isso acontece pelo menos em
    `tipo_localizacao_diferenciada` (códigos 0 e 8 ainda sem tradução na
    fatia do dicionário que conferi), fica visível no dado em vez de
    escondido.
    """
    df = df.copy()
    subset = dicionario[dicionario["id_tabela"] == id_tabela]
    colunas_alvo = colunas if colunas is not None else sorted(subset["nome_coluna"].unique().tolist())

    for coluna in colunas_alvo:
        if coluna not in df.columns:
            continue

        chaves_brutas = subset.loc[subset["nome_coluna"] == coluna, "chave"]
        valores = subset.loc[subset["nome_coluna"] == coluna, "valor"]
        mapa = {_normalizar_chave_codigo(k): v for k, v in zip(chaves_brutas, valores)}

        def _traduzir(valor, mapa=mapa):
            if pd.isna(valor):
                return "desconhecido"
            chave = _normalizar_chave_codigo(valor)
            return mapa.get(chave, f"{VALOR_CODIGO_NAO_DOCUMENTADO}_{chave}")

        df[coluna] = df[coluna].map(_traduzir)

    return df


def montar_base_escola_silver(ano: int = ANO_ESCOLHIDO) -> pd.DataFrame:
    """
    Monta a base "silver" de escola: aplica os quatro filtros já
    decididos (crosswalk com escola_completo, rede pública, anos
    iniciais, ano) e junta o IDEB com as 455 colunas da escola_completo,
    já com a variável-alvo binária calculada.

    Recebe `ano` como parâmetro (em vez de fixo) pra eu poder comparar
    facilmente com outro ciclo do IDEB no futuro, se um dia precisar -
    mas o valor padrão é a decisão final (2025), documentada em
    reports/decisoes.md.
    """
    ideb_escola = ler_ideb_escola()
    escola_completo = ler_escola_completo()

    ideb_escola = ideb_escola.copy()
    escola_completo = escola_completo.copy()
    ideb_escola["id_escola"] = ideb_escola["id_escola"].astype(str)
    escola_completo["id_escola"] = escola_completo["id_escola"].astype(str)

    # crosswalk validado no notebook 06 (Achado 14 em reports/decisoes.md)
    # - diferente do que aconteceu entre `alunos` e `escola` (Achado 11),
    # esses dois `id_escola` batem de verdade (95,89% de interseção)
    interseccao = set(ideb_escola["id_escola"].unique()) & set(escola_completo["id_escola"].unique())

    ideb_filtrado = ideb_escola[
        ideb_escola["id_escola"].isin(interseccao)
        & (ideb_escola["rede"].astype(str).str.lower() != "privada")
        & (ideb_escola["anos_escolares"] == "iniciais (1-5)")
        & (ideb_escola["ano"] == ano)
    ].copy()

    print(f"Linhas em ideb_filtrado (censo + pública + anos iniciais + ano {ano}): {len(ideb_filtrado):,}")

    duplicadas = ideb_filtrado["id_escola"].duplicated().sum()
    if duplicadas:
        print(f"Atenção: {duplicadas} id_escola duplicados nesse recorte - mantendo a primeira ocorrência.")
        ideb_filtrado = ideb_filtrado.drop_duplicates(subset="id_escola", keep="first")

    colunas_alvo = ["id_escola", "id_municipio", "sigla_uf", "rede", "taxa_aprovacao", "ideb"]
    base = ideb_filtrado[colunas_alvo].merge(
        escola_completo, on="id_escola", how="left", suffixes=("", "_censo")
    )

    # variável-alvo binária, decisão final registrada em reports/decisoes.md
    # (corte = meta nacional de referência do MEC/INEP, não um valor
    # escolhido arbitrariamente)
    base["alvo_ideb"] = (base["ideb"] >= 6.0).astype("Int64")
    base.loc[base["ideb"].isna(), "alvo_ideb"] = pd.NA

    print(f"\nbase_escola_silver final: {base.shape[0]:,} escolas, {base.shape[1]} colunas")
    print(f"% com 'ideb' preenchido: {base['ideb'].notna().mean() * 100:.2f}%")
    print(f"% com 'alvo_ideb' definido: {base['alvo_ideb'].notna().mean() * 100:.2f}%")

    # mesma rede de segurança que já uso em load_data.py: procuro qualquer
    # coluna que componha o próprio IDEB e que eu não tenha listado ainda
    # em COLUNAS_ALVO_EXCLUIR_DE_FEATURES - se aparecer aqui fora da lista
    # esperada, preciso revisar antes de usar como feature
    suspeitas = [
        c for c in base.columns
        if any(termo in c.lower() for termo in ["ideb", "saeb", "taxa_aprovacao", "indicador_rendimento", "projecao"])
        and c not in COLUNAS_ALVO_EXCLUIR_DE_FEATURES + ["alvo_ideb"]
    ]
    if suspeitas:
        print(f"\n⚠️  ATENÇÃO: colunas suspeitas de leakage ainda na base: {suspeitas}")
        print("Revisar antes de usar essas colunas como feature.")

    return base


def obter_base_escola_modelagem(forcar_releitura: bool = False) -> pd.DataFrame:
    """
    Retorna a base "silver" de escola, usando um cache local em parquet
    quando disponível - mesmo padrão que já uso em
    `load_data.obter_base_modelagem()` pra base de aluno.

    forcar_releitura=True ignora o cache e remonta a base do zero
    (útil se eu mudar algum dos filtros/decisões acima).
    """
    if CACHE_BASE_ESCOLA_MODELAGEM.exists() and not forcar_releitura:
        print(f"Lendo base_escola_modelagem do cache local: {CACHE_BASE_ESCOLA_MODELAGEM}")
        return pd.read_parquet(CACHE_BASE_ESCOLA_MODELAGEM)

    base = montar_base_escola_silver()
    CACHE_BASE_ESCOLA_MODELAGEM.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(CACHE_BASE_ESCOLA_MODELAGEM, index=False)
    print(f"Cache salvo em: {CACHE_BASE_ESCOLA_MODELAGEM}")
    return base


def publicar_base_escola_modelo(forcar_releitura: bool = False) -> pd.DataFrame:
    """
    Monta a base de escola pronta pra modelagem e já com as colunas
    categóricas traduzidas (código -> texto, via `traduzir_categoricas_escola`)
    e publica o resultado como uma camada nova no S3, `silver_modelo` -
    decisão registrada em `reports/decisoes.md`: a junção com o dicionário
    deixa de ser um passo dentro do notebook (EDA/seleção de features) e
    passa a fazer parte do pré-processamento, pronta assim que qualquer
    notebook pedir a base via `ler_base_escola_modelo()`.

    Rodar isso de novo só é necessário se eu mudar alguma decisão de
    filtro/tradução rio acima (ex.: um novo Achado sobre o dicionário) - no
    dia a dia, os notebooks usam `ler_base_escola_modelo()`, que lê o
    resultado já publicado aqui em vez de remontar tudo de novo.

    Bug real encontrado testando contra a base de verdade: não posso passar
    `colunas=None` pro `traduzir_categoricas_escola` aqui, porque o
    dicionário documenta "rede" como coluna categórica da própria
    `escola_completo` (códigos 1-4) - mas a `rede` que sobrevive em `base`
    vem do lado do IDEB, já como texto ("municipal"/"estadual"/"federal").
    Traduzir de novo por engano vira "codigo_nao_documentado_municipal".
    Por isso restrinjo explicitamente as colunas traduzidas, excluindo as de
    identificação/alvo que colidem de nome com o lado já-texto do IDEB.
    """
    base = obter_base_escola_modelagem(forcar_releitura=forcar_releitura)
    dicionario = ler_dicionario_escola(forcar_releitura=forcar_releitura)

    # Bug real que só apareceu testando contra a base de verdade: o
    # dicionário documenta "rede" (coluna categórica da própria
    # `escola_completo`, códigos 1-4), mas a coluna `rede` que sobrevive
    # aqui em `base` vem do lado do IDEB, já como texto ("municipal",
    # "estadual", "federal") - não é o mesmo código. Traduzir ela de novo
    # tentava converter texto pra número e caía no fallback
    # "codigo_nao_documentado_<valor>". A cópia vinda de `escola_completo`
    # (numérica de verdade) tem sufixo `_censo` no merge - essa sim pode ser
    # traduzida sem problema. Por isso excluo daqui as colunas de
    # identificação/alvo que colidem de nome com o lado já-texto do IDEB.
    COLUNAS_NAO_TRADUZIR = {"id_escola", "id_municipio", "sigla_uf", "rede", "ideb", "alvo_ideb"}
    colunas_categoricas = [
        c for c in colunas_categoricas_com_dicionario(dicionario)
        if c in base.columns and c not in COLUNAS_NAO_TRADUZIR and not c.startswith("id_")
    ]
    base_traduzida = traduzir_categoricas_escola(base, dicionario, colunas=colunas_categoricas)

    CACHE_BASE_ESCOLA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    base_traduzida.to_parquet(CACHE_BASE_ESCOLA_MODELO, index=False)
    print(f"Cache local salvo em: {CACHE_BASE_ESCOLA_MODELO}")

    nome_arquivo = f"base_escola_modelo_ano={ANO_ESCOLHIDO}.parquet"
    _salvar_parquet_no_prefixo(base_traduzida, BUCKET, PREFIXO_SILVER_MODELO_BASE_ESCOLA, nome_arquivo)

    return base_traduzida


def ler_base_escola_modelo(forcar_releitura: bool = False) -> pd.DataFrame:
    """
    Lê a base de escola já pronta pra modelagem - grão de escola, alvo
    calculado, colunas categóricas já traduzidas pra texto pelo dicionário
    oficial do INEP. Esta é a função que os notebooks (07, 08) devem usar a
    partir de agora, em vez de chamar `obter_base_escola_modelagem()` e
    traduzir o dicionário dentro do próprio notebook - decisão registrada em
    `reports/decisoes.md`: a junção com o dicionário sai do notebook e vira
    parte do pré-processamento.

    Ordem de tentativa: cache local -> camada `silver_modelo` já publicada
    no S3 -> se nada disso existir ainda (primeira vez rodando depois dessa
    mudança, ou num ambiente novo), monta e publica a camada na hora.
    """
    if CACHE_BASE_ESCOLA_MODELO.exists() and not forcar_releitura:
        print(f"Lendo base_escola_modelo do cache local: {CACHE_BASE_ESCOLA_MODELO}")
        return pd.read_parquet(CACHE_BASE_ESCOLA_MODELO)

    if not forcar_releitura:
        try:
            print(f"Cache local não encontrado, lendo camada silver_modelo do S3: "
                  f"s3://{BUCKET}/{PREFIXO_SILVER_MODELO_BASE_ESCOLA}")
            df = _ler_parquet_do_prefixo(BUCKET, PREFIXO_SILVER_MODELO_BASE_ESCOLA)
            CACHE_BASE_ESCOLA_MODELO.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(CACHE_BASE_ESCOLA_MODELO, index=False)
            print(f"Cache local salvo em: {CACHE_BASE_ESCOLA_MODELO}")
            return df
        except FileNotFoundError:
            print("Camada silver_modelo ainda não existe no S3 - montando e publicando agora.")

    return publicar_base_escola_modelo(forcar_releitura=forcar_releitura)


if __name__ == "__main__":
    # Rodar este script isoladamente serve só pra eu conferir que a base
    # de escola está sendo montada corretamente antes de seguir para a
    # seleção de features e a modelagem de fato.
    base = obter_base_escola_modelagem(forcar_releitura=True)
    print(base.head())
    print(base.dtypes)

    # também confiro a camada já traduzida/publicada (silver_modelo),
    # que é a que os notebooks de EDA/seleção de features passam a usar
    base_modelo = publicar_base_escola_modelo(forcar_releitura=True)
    print(base_modelo.head())
