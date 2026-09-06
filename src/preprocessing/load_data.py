"""
Leitura dos dados da Fase 2 direto do S3.

A base de modelagem da Fase 3 precisa de granularidade por aluno, mas a
camada Gold da Fase 2 é toda agregada por município. Por isso a base
principal aqui é a Silver `alunos` (a única com 1 linha por aluno avaliado),
enriquecida depois com o contexto de infraestrutura escolar que já está
pronto na Gold. Essa decisão está documentada no handoff do projeto.

Uso boto3 puro (sem awswrangler) pra manter uma dependência a menos - baixo
o Parquet pra memória com boto3 e leio com pandas/pyarrow.

Os paths abaixo são os buckets reais da Fase 2 (Engenharia de Dados) -
não tenho permissão de escrita lá, só leitura.
"""

import io
import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BUCKET = "pos-tech-fiap-985034838182-us-east-1-an"

# Silver: granularidade por aluno - é a base do modelo
PREFIXO_SILVER_ALUNOS = "silver/alunos/"

# Gold: granularidade por município/ano - usada só para enriquecer o aluno
# com o contexto territorial da escola/município dele
PREFIXO_GOLD_INFRAESTRUTURA = "gold/indicador_x_infraestrutura_escolar/"


def _ler_parquet_do_prefixo(bucket: str, prefixo: str) -> pd.DataFrame:
    """
    Lista todos os arquivos .parquet dentro de um prefixo do S3 e concatena
    num DataFrame só. As tabelas da Fase 2 costumam vir particionadas em
    vários arquivos (um por partição Hive-style, ex: ano=2023/), então não
    dá pra assumir que é um único objeto.
    """
    s3 = boto3.client("s3")
    paginador = s3.get_paginator("list_objects_v2")

    partes = []
    for pagina in paginador.paginate(Bucket=bucket, Prefix=prefixo):
        for objeto in pagina.get("Contents", []):
            chave = objeto["Key"]
            if not chave.endswith(".parquet"):
                continue
            corpo = s3.get_object(Bucket=bucket, Key=chave)["Body"].read()
            df_parte = pd.read_parquet(io.BytesIO(corpo))

            # a coluna "ano" é uma partição Hive-style (ano=2023/arquivo.parquet)
            # e não existe fisicamente dentro do parquet - preciso recuperar
            # ela a partir do caminho, senão a coluna some (isso já é um
            # comportamento documentado no handoff da Fase 2)
            if "ano" not in df_parte.columns:
                for pedaco in chave.split("/"):
                    if pedaco.startswith("ano="):
                        df_parte["ano"] = int(pedaco.split("=")[1])
                        break

            partes.append(df_parte)

    if not partes:
        raise FileNotFoundError(f"Nenhum .parquet encontrado em s3://{bucket}/{prefixo}")

    return pd.concat(partes, ignore_index=True)


def ler_alunos() -> pd.DataFrame:
    """
    Lê a Silver `alunos` (1 linha por aluno avaliado, ~3,87M linhas).

    Essa é a tabela-base do modelo: é a única com granularidade de
    indivíduo, todas as outras (uf, municipio, gold) são agregadas.
    """
    df = _ler_parquet_do_prefixo(BUCKET, PREFIXO_SILVER_ALUNOS)
    print(f"alunos: {df.shape[0]:,} linhas, {df.shape[1]} colunas")
    return df


def ler_infraestrutura_municipio() -> pd.DataFrame:
    """
    Lê a Gold `indicador_x_infraestrutura_escolar` (1 linha por
    município/ano, já filtrada em rede='5' - Pública).

    Uso isso só para trazer o contexto de infraestrutura escolar
    (água, internet, laboratório etc.) pra cada aluno via JOIN em
    (id_municipio, ano). Não uso essa tabela como base porque ela não
    tem granularidade de aluno.
    """
    df = _ler_parquet_do_prefixo(BUCKET, PREFIXO_GOLD_INFRAESTRUTURA)
    print(f"infraestrutura por município: {df.shape[0]:,} linhas, {df.shape[1]} colunas")
    return df


def montar_base_modelagem() -> pd.DataFrame:
    """
    Junta alunos (Silver) com infraestrutura escolar (Gold) via
    (id_municipio, ano), trazendo contexto territorial pra cada aluno.

    Uso LEFT JOIN porque quero manter todos os alunos mesmo que o
    município deles não tenha correspondência na infraestrutura (o
    handoff já registra que isso acontece e não é erro).
    """
    alunos = ler_alunos()
    infra = ler_infraestrutura_municipio()

    # infra tem colunas repetidas de indicador_por_municipio (taxa_alfabetizacao,
    # media_portugues etc.) que não quero herdar aqui - essas métricas são
    # derivadas do próprio indicador que estou tentando prever, então
    # trazê-las seria vazamento. Fico só com as colunas de infraestrutura.
    colunas_infra = [
        "id_municipio",
        "ano",
        "pct_escolas_agua_potavel",
        "pct_escolas_internet",
        "pct_escolas_equipamento_computador",
        "pct_escolas_biblioteca",
        "pct_escolas_laboratorio_informatica",
        "pct_escolas_esgoto_rede_publica",
    ]
    infra_reduzida = infra[colunas_infra]

    base = alunos.merge(infra_reduzida, on=["id_municipio", "ano"], how="left")
    print(f"base final: {base.shape[0]:,} linhas, {base.shape[1]} colunas")
    return base


if __name__ == "__main__":
    # Rodar este script isoladamente serve só para eu conferir que a leitura
    # do S3 está funcionando e ver o tamanho real da base antes de seguir
    # para a análise exploratória.
    base = montar_base_modelagem()
    print(base.head())
    print(base.dtypes)
