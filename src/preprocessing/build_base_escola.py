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

import pandas as pd

from src.preprocessing.load_data import BUCKET, _ler_parquet_do_prefixo

# Decisão final, documentada em reports/decisoes.md (Achado 17): o Censo
# Escolar (escola_completo) só existe pra ano=2024, e o IDEB só é
# calculado em anos ímpares. Uso o ciclo de 2025 do IDEB pareado com a
# infraestrutura de 2024 - infraestrutura escolar muda pouco de um ano
# pro outro, então essa combinação é uma aproximação razoável.
ANO_ESCOLHIDO = 2025

PREFIXO_BRONZE_IDEB_ESCOLA = "bronze/br_inep_ideb/escola/"
PREFIXO_BRONZE_ESCOLA_COMPLETO = "bronze/br_inep_censo_escolar/escola_completo/"

# Mesmos caches locais já usados nos notebooks 06/07 - se eu já rodei
# aqueles notebooks antes, esses arquivos já existem e a leitura é
# instantânea (sem baixar do S3 de novo).
CACHE_IDEB_ESCOLA = Path(__file__).resolve().parents[2] / "data" / "processed" / "ideb_escola.parquet"
CACHE_ESCOLA_COMPLETO = Path(__file__).resolve().parents[2] / "data" / "processed" / "escola_completo.parquet"
CACHE_BASE_ESCOLA_MODELAGEM = Path(__file__).resolve().parents[2] / "data" / "processed" / "base_escola_modelagem.parquet"

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


if __name__ == "__main__":
    # Rodar este script isoladamente serve só pra eu conferir que a base
    # de escola está sendo montada corretamente antes de seguir para a
    # seleção de features e a modelagem de fato.
    base = obter_base_escola_modelagem(forcar_releitura=True)
    print(base.head())
    print(base.dtypes)
