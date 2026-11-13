# Databricks notebook source
# DBTITLE 1,importando as config
# MAGIC %run ./05_configuracao_pipeline

# COMMAND ----------


from datetime import datetime, timezone
from uuid import uuid4
import time

from pyspark.sql import functions as F

# ============================================================
# SABOR EM FOCO - PIPELINE DE DADOS
# FASE 2 - QUESTOES 2.2 E 2.3
# ============================================================


# ============================================================
# 1. PARAMETROS
# ============================================================

defaults = {
    "etapa": "preparar",
    "modo": "inferencia",
    "lote_id": "operacional_1",
    "ambiente": "dev",
    "job_run_id": "teste_manual_001",
    "simular_falha": "false"
}

for nome, valor in defaults.items():
    try:
        dbutils.widgets.get(nome)
    except Exception:
        dbutils.widgets.text(nome, valor)

params = {
    nome: dbutils.widgets.get(nome).strip()
    for nome in defaults
}

etapa = params["etapa"]
modo = params["modo"]
lote_id = params["lote_id"]
ambiente = params["ambiente"]
job_run_id = params["job_run_id"]

simular_falha = params["simular_falha"].lower() == "true"

tentativa_id = str(uuid4())
inicio = datetime.now(timezone.utc)
relogio = time.perf_counter()


# ============================================================
# 2. CONFIGURACAO DA FONTE E TABELAS
# ============================================================

TABELA_FONTE = f"{BASE_DEV}.gold_reviews_ml_deduplicada"
TABELA_PREPARADA = f"{BASE_OPERACAO}.dados_pipeline"
TABELA_LOG = TABELAS["logs_execucao"]

VERSAO_FONTE = 0

RECORTES_DEDUPLICADOS = {
    "desenvolvimento": 313222,
    "operacional_1": 40730,
    "operacional_2": 40681
}

ESQUEMA_LOG = """
    tentativa_id STRING,
    job_run_id STRING,
    lote_id STRING,
    ambiente STRING,
    modo STRING,
    etapa STRING,
    inicio TIMESTAMP,
    fim TIMESTAMP,
    duracao_segundos DOUBLE,
    status STRING,
    registros_entrada BIGINT,
    registros_saida BIGINT,
    registros_invalidos BIGINT,
    versao_dados BIGINT,
    erro STRING
"""

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TABELA_LOG} (
    {ESQUEMA_LOG}
)
USING DELTA
""")


# ============================================================
# 3. LOG OPERACIONAL
# ============================================================

def registrar_log(
    status,
    entrada=0,
    saida=0,
    invalidos=0,
    versao=None,
    erro=None,
    finalizar=False
):

    fim = datetime.now(timezone.utc) if finalizar else None

    duracao = (
        round(time.perf_counter() - relogio, 3)
        if finalizar else None
    )

    registro = [(
        tentativa_id,
        job_run_id,
        lote_id,
        ambiente,
        modo,
        etapa,
        inicio,
        fim,
        duracao,
        status,
        int(entrada),
        int(saida),
        int(invalidos),
        versao,
        erro
    )]

    df_log = spark.createDataFrame(registro, ESQUEMA_LOG)

    df_log.createOrReplaceTempView("vw_log_execucao")

    spark.sql(f"""
    MERGE INTO {TABELA_LOG} AS d
    USING vw_log_execucao AS o
    ON d.tentativa_id = o.tentativa_id

    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """)


# ============================================================
# 4. VALIDACAO DOS PARAMETROS
# ============================================================

def validar_parametros():

    if etapa not in {"preparar", "validar"}:
        raise ValueError("Etapa invalida.")

    if modo not in {
        "treinamento",
        "inferencia",
        "retreinamento"
    }:
        raise ValueError("Modo invalido.")

    if ambiente not in {"dev", "operacao"}:
        raise ValueError("Ambiente invalido.")

    if lote_id not in {"operacional_1", "operacional_2"}:
        raise ValueError("Lote invalido.")

    if not job_run_id:
        raise ValueError("job_run_id obrigatorio.")

    if modo == "treinamento" and lote_id != "operacional_1":
        raise ValueError(
            "Treinamento inicial deve utilizar operacional_1."
        )

    if modo == "retreinamento" and lote_id != "operacional_2":
        raise ValueError(
            "Retreinamento deve utilizar operacional_2."
        )

    if params["simular_falha"].lower() not in {"true", "false"}:
        raise ValueError("simular_falha deve ser true ou false.")


def obter_recortes():

    if modo == "treinamento":
        return ["desenvolvimento", "operacional_1"]

    if modo == "retreinamento":
        return [
            "desenvolvimento",
            "operacional_1",
            "operacional_2"
        ]

    return [lote_id]


# ============================================================
# 5. EXECUCAO DO PIPELINE
# ============================================================

entrada = 0
saida = 0
invalidos = 0
versao = VERSAO_FONTE

registrar_log("EM_EXECUCAO", versao=versao)

try:

    validar_parametros()

    recortes = obter_recortes()

    print("=== CONTEXTO DA EXECUCAO ===")
    print(f"Etapa: {etapa}")
    print(f"Modo: {modo}")
    print(f"Lote: {lote_id}")
    print(f"Ambiente: {ambiente}")
    print(f"Job run: {job_run_id}")
    print(f"Tentativa: {tentativa_id}")
    print(f"Fonte: {TABELA_FONTE}")
    print(f"Versao fonte: {versao}")
    print(f"Recortes: {recortes}")


    # ========================================================
    # 5.1 PREPARACAO DOS DADOS
    # ========================================================

    if etapa == "preparar":

        df_origem = (
            spark.read
            .option("versionAsOf", versao)
            .table(TABELA_FONTE)
            .filter(F.col("recorte_temporal").isin(recortes))
        )

        entrada = df_origem.count()

        df_preparado = (
            df_origem
            .select(
                "Id",
                "ProductId",
                "time_unix",
                "data_avaliacao",
                "texto_entrada",
                "alvo_baixa_nota",
                "recorte_temporal",
                "assinatura_textual"
            )
            .withColumn("job_run_id", F.lit(job_run_id))
            .withColumn("lote_execucao", F.lit(lote_id))
            .withColumn("modo_execucao", F.lit(modo))
            .withColumn("ambiente_execucao", F.lit(ambiente))
            .withColumn("versao_fonte", F.lit(versao).cast("long"))
        )

        (
            df_preparado.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(TABELA_PREPARADA)
        )

        saida = spark.table(TABELA_PREPARADA).count()

        if entrada != saida:
            raise ValueError(
                "Divergencia entre entrada e saida."
            )

        print("\n=== PREPARACAO CONCLUIDA ===")
        print(f"Tabela: {TABELA_PREPARADA}")
        print(f"Registros preparados: {saida:,}")

        display(
            spark.table(TABELA_PREPARADA)
            .groupBy("recorte_temporal")
            .count()
            .orderBy("recorte_temporal")
        )


    # ========================================================
    # 5.2 VALIDACAO BLOQUEANTE
    # ========================================================

    elif etapa == "validar":

        if not spark.catalog.tableExists(TABELA_PREPARADA):
            raise ValueError("Tabela preparada inexistente.")

        df = spark.table(TABELA_PREPARADA)

        # Simula uma falha critica sem alterar a tabela Delta
        if simular_falha:
            print("SIMULACAO DE FALHA: entrada vazia.")
            df = df.limit(0)

        entrada = df.count()

        if entrada == 0:
            raise ValueError(
                "FALHA CRITICA: tabela de entrada vazia."
            )

        # Confirma que os dados pertencem ao Job atual

        contexto = df.select(
            "job_run_id",
            "lote_execucao",
            "modo_execucao",
            "ambiente_execucao",
            "versao_fonte"
        ).distinct().collect()

        if len(contexto) != 1:
            raise ValueError("Contexto de execucao inconsistente.")

        contexto = contexto[0]

        if (
            contexto["job_run_id"] != job_run_id
            or contexto["lote_execucao"] != lote_id
            or contexto["modo_execucao"] != modo
            or contexto["ambiente_execucao"] != ambiente
            or contexto["versao_fonte"] != versao
        ):
            raise ValueError(
                "Dados pertencem a outra execucao."
            )

        # Regras criticas de qualidade

        invalidos = df.filter(
            F.col("Id").isNull()
            | F.col("ProductId").isNull()
            | (F.trim("ProductId") == "")
            | F.col("time_unix").isNull()
            | F.col("data_avaliacao").isNull()
            | F.col("texto_entrada").isNull()
            | (F.trim("texto_entrada") == "")
            | F.col("alvo_baixa_nota").isNull()
            | (~F.col("alvo_baixa_nota").isin(0, 1))
            | F.col("assinatura_textual").isNull()
            | F.col("recorte_temporal").isNull()
            | (~F.col("recorte_temporal").isin(recortes))
            | (
                F.col("recorte_temporal")
                != classificar_recorte(F.col("time_unix"))
            )
        ).count()

        duplicados_id = (
            entrada - df.select("Id").distinct().count()
        )

        duplicados_texto = (
            entrada
            - df.select("assinatura_textual").distinct().count()
        )

        contagens = {
            r["recorte_temporal"]: r["count"]
            for r in df.groupBy("recorte_temporal").count().collect()
        }

        esperado = {
            recorte: RECORTES_DEDUPLICADOS[recorte]
            for recorte in recortes
        }

        print("\n=== CONTROLE DE QUALIDADE ===")
        print(f"Registros analisados: {entrada:,}")
        print(f"Registros invalidos: {invalidos:,}")
        print(f"IDs duplicados: {duplicados_id:,}")
        print(f"Textos duplicados: {duplicados_texto:,}")
        print(f"Recortes encontrados: {contagens}")
        print(f"Recortes esperados: {esperado}")

        if invalidos > 0:
            raise ValueError(
                "FALHA CRITICA: registros invalidos."
            )

        if duplicados_id > 0:
            raise ValueError(
                "FALHA CRITICA: IDs duplicados."
            )

        if duplicados_texto > 0:
            raise ValueError(
                "FALHA CRITICA: assinaturas textuais duplicadas."
            )

        if contagens != esperado:
            raise ValueError(
                "FALHA CRITICA: contagens divergentes."
            )

        saida = entrada

        print("\nVALIDACAO APROVADA")
        print("Dados liberados para Machine Learning.")


    # ========================================================
    # 6. FINALIZACAO COM SUCESSO
    # ========================================================

    registrar_log(
        "SUCESSO",
        entrada=entrada,
        saida=saida,
        invalidos=invalidos,
        versao=versao,
        finalizar=True
    )

    print("\n=== RESULTADO FINAL ===")
    print(f"Etapa: {etapa}")
    print("Status: SUCESSO")
    print(f"Entrada: {entrada:,}")
    print(f"Saida: {saida:,}")
    print(f"Tentativa: {tentativa_id}")


except Exception as erro:

    # ========================================================
    # 7. REGISTRO DE FALHA
    # ========================================================

    registrar_log(
        "FALHA",
        entrada=entrada,
        saida=saida,
        invalidos=invalidos,
        versao=versao,
        erro=str(erro)[:4000],
        finalizar=True
    )

    print("\n=== PIPELINE INTERROMPIDO ===")
    print(f"Etapa: {etapa}")
    print(f"Erro: {erro}")

    # Propaga a falha ao Lakeflow Job
    raise


# ============================================================
# 8. EVIDENCIA OPERACIONAL
# ============================================================

print("\n=== LOG OPERACIONAL ===")

display(
    spark.table(TABELA_LOG)
    .filter(F.col("tentativa_id") == tentativa_id)
)


# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     job_run_id,
# MAGIC     tentativa_id,
# MAGIC     lote_id,
# MAGIC     etapa,
# MAGIC     inicio,
# MAGIC     fim,
# MAGIC     duracao_segundos,
# MAGIC     status,
# MAGIC     registros_entrada,
# MAGIC     registros_saida,
# MAGIC     registros_invalidos,
# MAGIC     versao_dados,
# MAGIC     erro
# MAGIC
# MAGIC FROM workspace.sabor_operacao.logs_execucao
# MAGIC
# MAGIC WHERE job_run_id NOT LIKE 'teste_manual%'
# MAGIC
# MAGIC ORDER BY inicio DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     job_run_id,
# MAGIC     lote_id,
# MAGIC     COUNT(*) AS total_etapas,
# MAGIC
# MAGIC     COUNT_IF(status = 'SUCESSO')
# MAGIC         AS etapas_sucesso,
# MAGIC
# MAGIC     COUNT_IF(status = 'FALHA')
# MAGIC         AS etapas_falha,
# MAGIC
# MAGIC     ROUND(SUM(duracao_segundos), 2)
# MAGIC         AS duracao_total_segundos,
# MAGIC
# MAGIC     MIN(inicio) AS inicio_execucao,
# MAGIC     MAX(fim) AS fim_execucao
# MAGIC
# MAGIC FROM workspace.sabor_operacao.logs_execucao
# MAGIC
# MAGIC WHERE job_run_id NOT LIKE 'teste_manual%'
# MAGIC
# MAGIC GROUP BY job_run_id, lote_id
# MAGIC
# MAGIC ORDER BY inicio_execucao DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     job_run_id,
# MAGIC     lote_id,
# MAGIC     COUNT(*) AS total_etapas,
# MAGIC     COUNT_IF(status = 'SUCESSO') AS etapas_sucesso,
# MAGIC     COUNT_IF(status = 'FALHA') AS etapas_falha,
# MAGIC     ROUND(SUM(duracao_segundos), 2) AS duracao_segundos
# MAGIC FROM workspace.sabor_operacao.logs_execucao
# MAGIC WHERE job_run_id NOT LIKE 'teste_manual%'
# MAGIC GROUP BY job_run_id, lote_id
# MAGIC ORDER BY MAX(inicio) DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     COUNT(*) AS total,
# MAGIC     COUNT(DISTINCT Id) AS ids_distintos,
# MAGIC     COUNT(*) - COUNT(DISTINCT Id) AS duplicidades
# MAGIC FROM workspace.sabor_operacao.predicoes_reviews
# MAGIC WHERE lote_id = 'operacional_2'
# MAGIC   AND versao_modelo = '3';