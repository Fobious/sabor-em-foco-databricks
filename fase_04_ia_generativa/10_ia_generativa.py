# Databricks notebook source
# MAGIC
# MAGIC %sql
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS
# MAGIC workspace.sabor_operacao.amostra_ia
# MAGIC USING DELTA
# MAGIC AS
# MAGIC
# MAGIC SELECT
# MAGIC     Id,
# MAGIC     recorte_temporal AS lote_id,
# MAGIC     alvo_baixa_nota,
# MAGIC     texto_entrada AS texto_original,
# MAGIC     LEFT(texto_entrada, 4000) AS texto_ia,
# MAGIC     LENGTH(texto_entrada) AS caracteres_originais,
# MAGIC     LENGTH(texto_entrada) > 4000 AS texto_truncado
# MAGIC
# MAGIC FROM workspace.sabor_dev.gold_reviews_ml_deduplicada
# MAGIC
# MAGIC WHERE recorte_temporal = 'operacional_1'
# MAGIC
# MAGIC ORDER BY XXHASH64(Id, 42), Id
# MAGIC
# MAGIC LIMIT 50;
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS
# MAGIC workspace.sabor_operacao.resultados_ia (
# MAGIC
# MAGIC     Id BIGINT,
# MAGIC     lote_id STRING,
# MAGIC     funcao STRING,
# MAGIC     parametros STRING,
# MAGIC     texto_entrada STRING,
# MAGIC     texto_truncado BOOLEAN,
# MAGIC     resultado STRING,
# MAGIC     status STRING,
# MAGIC     erro STRING,
# MAGIC     instante_execucao TIMESTAMP,
# MAGIC     duracao_segundos DOUBLE
# MAGIC
# MAGIC )
# MAGIC USING DELTA;
# MAGIC

# COMMAND ----------

# DBTITLE 1,4.1-analise de sentimento

import time
from datetime import datetime, timezone
# questão 4.1 - ANALISE DE SENTIMENTO
# Começa com 3 avaliações para validar a função
LIMITE = 50

AMOSTRA = "workspace.sabor_operacao.amostra_ia"
DESTINO = "workspace.sabor_operacao.resultados_ia"

FUNCAO = "ai_analyze_sentiment"
PARAMETROS = '{"max_caracteres":4000}'

# 1. Seleciona somente avaliações ainda não processadas

pendentes = spark.sql(f"""
    SELECT
        a.Id,
        a.lote_id,
        a.texto_ia,
        a.texto_truncado

    FROM {AMOSTRA} a

    LEFT ANTI JOIN {DESTINO} r
        ON a.Id = r.Id
        AND a.lote_id = r.lote_id
        AND r.funcao = '{FUNCAO}'
        AND r.parametros = '{PARAMETROS}'

    ORDER BY XXHASH64(a.Id, 42), a.Id

    LIMIT {LIMITE}
""").collect()

# 2. Executa a AI Function exclusivamente via SQL

resultados = []

for linha in pendentes:

    inicio = time.perf_counter()

    resultado = None
    erro = None

    try:

        resultado = spark.sql(
            """
            SELECT ai_analyze_sentiment(:texto) AS sentimento
            """,
            args={"texto": linha["texto_ia"]}
        ).first()["sentimento"]

        status = (
            "NULO" if resultado is None
            else "SUCESSO"
        )

    except Exception as e:

        status = "FALHA"
        erro = str(e)[:4000]

    resultados.append((
        int(linha["Id"]),
        linha["lote_id"],
        FUNCAO,
        PARAMETROS,
        linha["texto_ia"],
        bool(linha["texto_truncado"]),
        resultado,
        status,
        erro,
        datetime.now(timezone.utc),
        round(time.perf_counter() - inicio, 3)
    ))

# 3. Persiste os resultados

if resultados:

    schema = spark.table(DESTINO).schema

    df = spark.createDataFrame(resultados, schema)

    df.createOrReplaceTempView("vw_resultados_ia")

    spark.sql(f"""
        MERGE INTO {DESTINO} AS d
        USING vw_resultados_ia AS o

        ON d.Id = o.Id
        AND d.lote_id = o.lote_id
        AND d.funcao = o.funcao
        AND d.parametros = o.parametros

        WHEN NOT MATCHED THEN INSERT *
    """)

# 4. Exibe os resultados armazenados

print("=== ANALISE DE SENTIMENTO ===")
print(f"Textos submetidos nesta execução: {len(pendentes)}")

display(
    spark.sql(f"""
        SELECT
            Id,
            resultado AS sentimento,
            status,
            texto_truncado,
            duracao_segundos,
            erro

        FROM {DESTINO}

        WHERE funcao = '{FUNCAO}'

        ORDER BY Id
    """)
)


# COMMAND ----------

# DBTITLE 1,divisão sentimentos
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     funcao,
# MAGIC     status,
# MAGIC     resultado,
# MAGIC     COUNT(*) AS quantidade
# MAGIC
# MAGIC FROM workspace.sabor_operacao.resultados_ia
# MAGIC
# MAGIC WHERE funcao = 'ai_analyze_sentiment'
# MAGIC
# MAGIC GROUP BY funcao, status, resultado
# MAGIC
# MAGIC ORDER BY status, resultado;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_processado,
# MAGIC     COUNT(DISTINCT Id) AS ids_distintos,
# MAGIC     COUNT_IF(status = 'SUCESSO') AS sucessos,
# MAGIC     COUNT_IF(status = 'FALHA') AS falhas,
# MAGIC     COUNT_IF(status = 'NULO') AS resultados_nulos
# MAGIC
# MAGIC FROM workspace.sabor_operacao.resultados_ia
# MAGIC
# MAGIC WHERE funcao = 'ai_analyze_sentiment';

# COMMAND ----------

from pyspark.sql import functions as F
import time
import re
from datetime import datetime, timezone

# =====================================================
# 4.2 - RESUMO AUTOMATICO
# =====================================================

LIMITE = 10

AMOSTRA = "workspace.sabor_operacao.amostra_ia"
DESTINO = "workspace.sabor_operacao.resultados_ia"

FUNCAO = "ai_summarize"
PARAMETROS = '{"max_words":40}'

# 1. Selecionar somente textos ainda nao resumidos

pendentes = spark.sql(f"""
    SELECT
        a.Id,
        a.lote_id,
        a.texto_ia,
        a.texto_truncado

    FROM {AMOSTRA} a

    LEFT ANTI JOIN {DESTINO} r
        ON a.Id = r.Id
        AND a.lote_id = r.lote_id
        AND r.funcao = '{FUNCAO}'
        AND r.parametros = '{PARAMETROS}'

    ORDER BY XXHASH64(a.Id, 42), a.Id

    LIMIT {LIMITE}
""").collect()

# 2. Executar ai_summarize

resultados = []

for linha in pendentes:

    inicio = time.perf_counter()

    resumo = None
    erro = None

    try:

        resumo = spark.sql(
            """
            SELECT ai_summarize(:texto, 40) AS resumo
            """,
            args={"texto": linha["texto_ia"]}
        ).first()["resumo"]

        status = "SUCESSO" if resumo else "NULO"

    except Exception as e:

        status = "FALHA"
        erro = str(e)[:4000]

    resultados.append((
        int(linha["Id"]),
        linha["lote_id"],
        FUNCAO,
        PARAMETROS,
        linha["texto_ia"],
        bool(linha["texto_truncado"]),
        resumo,
        status,
        erro,
        datetime.now(timezone.utc),
        round(time.perf_counter() - inicio, 3)
    ))

# 3. Persistir os resultados

if resultados:

    schema = spark.table(DESTINO).schema

    df = spark.createDataFrame(resultados, schema)

    df.createOrReplaceTempView("vw_resumos_ia")

    spark.sql(f"""
        MERGE INTO {DESTINO} AS d
        USING vw_resumos_ia AS o

        ON d.Id = o.Id
        AND d.lote_id = o.lote_id
        AND d.funcao = o.funcao
        AND d.parametros = o.parametros

        WHEN NOT MATCHED THEN INSERT *
    """)

# 4. Apresentar resultados e quantidade de palavras

print("=== RESUMOS AUTOMATICOS ===")
print(f"Textos submetidos: {len(pendentes)}")

display(
    spark.table(DESTINO)
    .filter(F.col("funcao") == FUNCAO)
    .withColumn(
        "quantidade_palavras",
        F.size(
            F.split(F.trim(F.col("resultado")), r"\s+")
        )
    )
    .select(
        "Id",
        "resultado",
        "quantidade_palavras",
        "status",
        "duracao_segundos",
        "erro"
    )
    .orderBy("Id")
)


# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     funcao,
# MAGIC     status,
# MAGIC     COUNT(*) AS quantidade,
# MAGIC     COUNT(DISTINCT Id) AS ids_distintos,
# MAGIC     MAX(
# MAGIC         SIZE(SPLIT(TRIM(resultado), '\\s+'))
# MAGIC     ) AS maior_resumo_palavras
# MAGIC
# MAGIC FROM workspace.sabor_operacao.resultados_ia
# MAGIC
# MAGIC WHERE funcao = 'ai_summarize'
# MAGIC
# MAGIC GROUP BY funcao, status;
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC
# MAGIC WITH resumos AS (
# MAGIC     SELECT
# MAGIC         Id,
# MAGIC         resultado,
# MAGIC         SIZE(
# MAGIC             SPLIT(TRIM(resultado), '\\s+')
# MAGIC         ) AS quantidade_palavras,
# MAGIC         status,
# MAGIC         ROW_NUMBER() OVER (
# MAGIC             ORDER BY Id
# MAGIC         ) AS ordem
# MAGIC     FROM workspace.sabor_operacao.resultados_ia
# MAGIC     WHERE funcao = 'ai_summarize'
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     Id,
# MAGIC     resultado,
# MAGIC     quantidade_palavras,
# MAGIC     status
# MAGIC FROM resumos
# MAGIC WHERE ordem <= 10
# MAGIC ORDER BY Id;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC questão 4.3
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC
# MAGIC WITH sentimentos AS (
# MAGIC     SELECT
# MAGIC         Id,
# MAGIC         resultado AS sentimento
# MAGIC     FROM workspace.sabor_operacao.resultados_ia
# MAGIC     WHERE funcao = 'ai_analyze_sentiment'
# MAGIC       AND status = 'SUCESSO'
# MAGIC ),
# MAGIC
# MAGIC resumos AS (
# MAGIC     SELECT
# MAGIC         Id,
# MAGIC         resultado AS resumo
# MAGIC     FROM workspace.sabor_operacao.resultados_ia
# MAGIC     WHERE funcao = 'ai_summarize'
# MAGIC       AND status = 'SUCESSO'
# MAGIC ),
# MAGIC
# MAGIC predicoes AS (
# MAGIC     SELECT *
# MAGIC     FROM (
# MAGIC         SELECT
# MAGIC             Id,
# MAGIC             classe_predita,
# MAGIC             versao_modelo,
# MAGIC             ROW_NUMBER() OVER (
# MAGIC                 PARTITION BY Id
# MAGIC                 ORDER BY instante_predicao DESC
# MAGIC             ) AS ordem
# MAGIC         FROM workspace.sabor_operacao.predicoes_reviews
# MAGIC         WHERE lote_id = 'operacional_1'
# MAGIC           AND CAST(versao_modelo AS STRING) = '2'
# MAGIC     )
# MAGIC     WHERE ordem = 1
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_amostra,
# MAGIC
# MAGIC     COUNT_IF(b.Score IS NOT NULL)
# MAGIC         AS com_nota_historica,
# MAGIC
# MAGIC     COUNT_IF(p.classe_predita IS NOT NULL)
# MAGIC         AS com_predicao_modelo,
# MAGIC
# MAGIC     COUNT_IF(s.sentimento IS NOT NULL)
# MAGIC         AS com_sentimento_ia,
# MAGIC
# MAGIC     COUNT_IF(r.resumo IS NOT NULL)
# MAGIC         AS com_resumo_ia,
# MAGIC
# MAGIC     COUNT_IF(
# MAGIC         b.Score IS NOT NULL
# MAGIC         AND p.classe_predita IS NOT NULL
# MAGIC         AND s.sentimento IS NOT NULL
# MAGIC         AND r.resumo IS NOT NULL
# MAGIC     ) AS comparacoes_completas,
# MAGIC
# MAGIC     COUNT_IF(a.texto_truncado)
# MAGIC         AS textos_truncados
# MAGIC
# MAGIC FROM workspace.sabor_operacao.amostra_ia a
# MAGIC
# MAGIC LEFT JOIN workspace.sabor_dev.bronze_reviews b
# MAGIC     ON a.Id = b.Id
# MAGIC
# MAGIC LEFT JOIN predicoes p
# MAGIC     ON a.Id = p.Id
# MAGIC
# MAGIC LEFT JOIN sentimentos s
# MAGIC     ON a.Id = s.Id
# MAGIC
# MAGIC LEFT JOIN resumos r
# MAGIC     ON a.Id = r.Id;
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     a.Id,
# MAGIC
# MAGIC     CAST(b.Score AS INT) AS nota_historica,
# MAGIC
# MAGIC     a.alvo_baixa_nota AS classe_real,
# MAGIC
# MAGIC     p.classe_predita,
# MAGIC
# MAGIC     ROUND(p.probabilidade_classe_1, 4)
# MAGIC         AS probabilidade_baixa_nota,
# MAGIC
# MAGIC     s.resultado AS sentimento_ia,
# MAGIC
# MAGIC     r.resultado AS resumo_ia,
# MAGIC
# MAGIC     a.texto_original,
# MAGIC
# MAGIC     CASE
# MAGIC         WHEN a.alvo_baixa_nota = p.classe_predita
# MAGIC         THEN 'ACERTO'
# MAGIC         ELSE 'ERRO'
# MAGIC     END AS avaliacao_modelo
# MAGIC
# MAGIC FROM workspace.sabor_operacao.amostra_ia a
# MAGIC
# MAGIC JOIN workspace.sabor_dev.bronze_reviews b
# MAGIC     ON a.Id = CAST(b.Id AS BIGINT)
# MAGIC
# MAGIC JOIN workspace.sabor_operacao.predicoes_reviews p
# MAGIC     ON a.Id = p.Id
# MAGIC     AND p.lote_id = 'operacional_1'
# MAGIC     AND p.versao_modelo = '2'
# MAGIC
# MAGIC JOIN workspace.sabor_operacao.resultados_ia s
# MAGIC     ON a.Id = s.Id
# MAGIC     AND s.funcao = 'ai_analyze_sentiment'
# MAGIC     AND s.status = 'SUCESSO'
# MAGIC
# MAGIC JOIN workspace.sabor_operacao.resultados_ia r
# MAGIC     ON a.Id = r.Id
# MAGIC     AND r.funcao = 'ai_summarize'
# MAGIC     AND r.status = 'SUCESSO'
# MAGIC
# MAGIC WHERE a.Id IN (
# MAGIC     11131,
# MAGIC     60771,
# MAGIC     63306,
# MAGIC     74608,
# MAGIC     103202,
# MAGIC     106255,
# MAGIC     120715,
# MAGIC     148000,
# MAGIC     213245,
# MAGIC     225841
# MAGIC )
# MAGIC
# MAGIC ORDER BY a.Id;
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     funcao,
# MAGIC
# MAGIC     COUNT(*) AS textos_submetidos,
# MAGIC
# MAGIC     COUNT(DISTINCT Id) AS ids_distintos,
# MAGIC
# MAGIC     MAX(LENGTH(texto_entrada))
# MAGIC         AS maior_entrada_caracteres,
# MAGIC
# MAGIC     COUNT_IF(LENGTH(texto_entrada) > 4000)
# MAGIC         AS entradas_acima_limite,
# MAGIC
# MAGIC     COUNT_IF(texto_truncado)
# MAGIC         AS textos_truncados,
# MAGIC
# MAGIC     COUNT_IF(status = 'SUCESSO')
# MAGIC         AS sucessos,
# MAGIC
# MAGIC     COUNT_IF(status = 'FALHA')
# MAGIC         AS falhas,
# MAGIC
# MAGIC     COUNT_IF(status = 'NULO')
# MAGIC         AS resultados_nulos,
# MAGIC
# MAGIC     CASE
# MAGIC         WHEN funcao = 'ai_summarize'
# MAGIC         THEN MAX(
# MAGIC             SIZE(SPLIT(TRIM(resultado), '\\s+'))
# MAGIC         )
# MAGIC         ELSE NULL
# MAGIC     END AS maior_resumo_palavras
# MAGIC
# MAGIC FROM workspace.sabor_operacao.resultados_ia
# MAGIC
# MAGIC GROUP BY funcao
# MAGIC
# MAGIC ORDER BY funcao;
# MAGIC