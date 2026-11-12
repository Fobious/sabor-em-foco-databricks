# Databricks notebook source
# DBTITLE 1,Preparação do alvo binário

from pyspark.sql import functions as F

# 1.3.1.a - Preparação inicial da camada Silver

TABELA_BRONZE = "workspace.sabor_dev.bronze_reviews"

# Fronteiras temporais definidas na Questão 1.1
CORTE_DESENVOLVIMENTO = 1336521600
CORTE_OPERACIONAL_1 = 1344384000

# Leitura da Bronze
df_bronze = spark.table(TABELA_BRONZE)

# Normalização dos campos numéricos e temporais
df_silver_preparado = (
    df_bronze
    .withColumn("Id", F.expr("try_cast(trim(Id) AS BIGINT)"))
    .withColumn("Score", F.expr("try_cast(trim(Score) AS INT)"))
    .withColumn(
        "HelpfulnessNumerator",
        F.expr("try_cast(HelpfulnessNumerator AS BIGINT)")
    )
    .withColumn(
        "HelpfulnessDenominator",
        F.expr("try_cast(HelpfulnessDenominator AS BIGINT)")
    )
    .withColumn(
        "time_unix",
        F.expr("try_cast(trim(Time) AS BIGINT)")
    )
    .withColumn(
        "data_avaliacao",
        F.to_date(F.timestamp_seconds("time_unix"))
    )
)

# Remove marcações HTML e normaliza espaços
for coluna in ["Summary", "Text"]:
    df_silver_preparado = df_silver_preparado.withColumn(
        f"{coluna}_limpo",
        F.trim(
            F.regexp_replace(
                F.regexp_replace(
                    F.coalesce(F.col(coluna), F.lit("")),
                    r"<[^>]*>",
                    " "
                ),
                r"\s+",
                " "
            )
        )
    )

# Combina Summary e Text para uso futuro no ML
df_silver_preparado = df_silver_preparado.withColumn(
    "texto_entrada",
    F.trim(
        F.concat_ws(
            " ",
            F.col("Summary_limpo"),
            F.col("Text_limpo")
        )
    )
)

# Identifica o recorte cronológico
df_silver_preparado = df_silver_preparado.withColumn(
    "recorte_temporal",
    F.when(
        F.col("time_unix") <= CORTE_DESENVOLVIMENTO,
        "desenvolvimento"
    )
    .when(
        F.col("time_unix") <= CORTE_OPERACIONAL_1,
        "operacional_1"
    )
    .otherwise("operacional_2")
)

print("=== 1.3.1.a - ESQUEMA SILVER ===")
df_silver_preparado.printSchema()

print("=== AMOSTRA DOS DADOS TRATADOS ===")
display(
    df_silver_preparado.select(
        "Id",
        "Score",
        "data_avaliacao",
        "Summary_limpo",
        F.substring("Text_limpo", 1, 100).alias("Text_inicio"),
        F.substring("texto_entrada", 1, 150).alias("entrada_inicio"),
        "recorte_temporal"
    ).limit(5)
)


# COMMAND ----------

# DBTITLE 1,Criação do alvo binário

from pyspark.sql import functions as F

# 1.3.1.b - Criação do alvo binário

df_silver_preparado = df_silver_preparado.withColumn(
    "alvo_baixa_nota",
    F.when(
        F.col("Score").isin(1, 2),
        F.lit(1)
    )
    .when(
        F.col("Score").isin(3, 4, 5),
        F.lit(0)
    )
    .otherwise(F.lit(None).cast("int"))
)

# Distribuição das avaliações p/ classe
distribuicao_alvo = (
    df_silver_preparado
    .groupBy("alvo_baixa_nota")
    .agg(F.count("*").alias("quantidade"))
    .orderBy("alvo_baixa_nota")
)

print("=== 1.3.1.b - DISTRIBUIÇÃO DO ALVO ===")
display(distribuicao_alvo)

# as colunas que serão utilizadas pelo modelo
print("=== ENTRADAS DO MODELO ===")
print("Preditor textual: texto_entrada")
print("Variável alvo: alvo_baixa_nota")
print("Score: utilizado apenas para gerar o alvo e avaliar posteriormente")


# COMMAND ----------

# DBTITLE 1,Preparação Merge

from pyspark.sql import functions as F
# 1.3.2.a - Fazendo a conferência dos dados antes da carga incremental
resumo_carga_silver = (
    df_silver_preparado
    .groupBy("recorte_temporal", "alvo_baixa_nota")
    .agg(F.count("*").alias("quantidade"))
    .orderBy("recorte_temporal", "alvo_baixa_nota")
)
print("=== DISTRIBUIÇÃO POR RECORTE E CLASSE ===")
display(resumo_carga_silver)

# COMMAND ----------

# DBTITLE 1,Carga inicial da Silver

# 1.3.2.a - Carga inicial da Silver
from pyspark.sql import functions as F
TABELA_SILVER = "workspace.sabor_dev.silver_reviews"

# puxa so período de desenvolvimento
df_desenvolvimento = (
    df_silver_preparado
    .filter(F.col("recorte_temporal") == "desenvolvimento")
    .select(
        "Id",
        "ProductId",
        "Score",
        "time_unix",
        "data_avaliacao",
        "Summary_limpo",
        "Text_limpo",
        "texto_entrada",
        "alvo_baixa_nota",
        "recorte_temporal",
        "arquivo_origem",
        "instante_ingestao",
        "lote_id"
    )
)

# Validação
total_entrada = df_desenvolvimento.count()

ids_distintos = (
    df_desenvolvimento.select("Id").distinct().count()
)

registros_invalidos = df_desenvolvimento.filter(
    F.col("Id").isNull()
    | F.col("ProductId").isNull()
    | (F.trim("ProductId") == "")
    | F.col("Score").isNull()
    | ~F.col("Score").between(1, 5)
    | F.col("data_avaliacao").isNull()
    | (F.trim("Text_limpo") == "")
    | F.col("alvo_baixa_nota").isNull()
).count()

print("=== VALIDAÇÃO DA CARGA INICIAL ===")
print(f"Registros de desenvolvimento: {total_entrada:,}")
print(f"IDs distintos: {ids_distintos:,}")
print(f"Registros inválidos: {registros_invalidos:,}")

if total_entrada != 454533:
    raise ValueError("Contagem diferente da referência temporal.")

if ids_distintos != total_entrada:
    raise ValueError("Foram identificados IDs duplicados.")

if registros_invalidos > 0:
    raise ValueError("Existem registros inválidos na carga.")

# Gravando a Silver
if not spark.catalog.tableExists(TABELA_SILVER):
    (
        df_desenvolvimento.write
        .format("delta")
        .mode("ignore")
        .saveAsTable(TABELA_SILVER)
    )
    print("Carga inicial da Silver solicitada.")
else:
    print("Silver já existe. Nenhuma gravação realizada.")

# Confere o conteúdo da Silver
df_silver = spark.table(TABELA_SILVER)

print("=== RESULTADO DA SILVER ===")
display(
    df_silver.groupBy("recorte_temporal")
    .agg(F.count("*").alias("quantidade"))
)


# COMMAND ----------

# DBTITLE 1,Merge incremental

# 1.3.2.a - MERGE incremental do primeiro lote operacional
from pyspark.sql import functions as F

TABELA_SILVER = "workspace.sabor_dev.silver_reviews"

# Recupera o primeiro lote a partir do DataFrame preparado
df_lote_operacional_1 = (
    df_silver_preparado
    .filter(F.col("recorte_temporal") == "operacional_1")
    .select(
        "Id",
        "ProductId",
        "Score",
        "time_unix",
        "data_avaliacao",
        "Summary_limpo",
        "Text_limpo",
        "texto_entrada",
        "alvo_baixa_nota",
        "recorte_temporal",
        "arquivo_origem",
        "instante_ingestao",
        "lote_id"
    )
)

# Disponibiliza os registros para o MERGE SQL
df_lote_operacional_1.createOrReplaceTempView(
    "vw_lote_operacional_1"
)

# Executa inserções somente para IDs que ainda não existem
spark.sql(f"""
MERGE INTO {TABELA_SILVER} AS destino
USING vw_lote_operacional_1 AS origem
ON destino.Id = origem.Id

WHEN NOT MATCHED THEN INSERT *
""")

print("=== MERGE DO LOTE OPERACIONAL 1 ===")

display(
    spark.table(TABELA_SILVER)
    .groupBy("recorte_temporal")
    .agg(F.count("*").alias("quantidade"))
    .orderBy("recorte_temporal")
)


# COMMAND ----------


from pyspark.sql import functions as F

# 1.3.2.a - Reexecução do lote operacional 1

TABELA_SILVER = "workspace.sabor_dev.silver_reviews"
TABELA_BRONZE = "workspace.sabor_dev.bronze_reviews"

CORTE_DESENVOLVIMENTO = 1336521600
CORTE_OPERACIONAL_1 = 1344384000

# Recria o lote a partir da Bronze, sem depender
# de variáveis de uma sessão anterior
df_lote_reexecucao = (
    spark.table(TABELA_BRONZE)
    .filter(
        (F.expr("try_cast(Time AS BIGINT)") > CORTE_DESENVOLVIMENTO) &
        (F.expr("try_cast(Time AS BIGINT)") <= CORTE_OPERACIONAL_1)
    )
    .select(
        F.expr("try_cast(Id AS BIGINT)").alias("Id")
    )
)

# Confere a quantidade de IDs do lote
total_lote = df_lote_reexecucao.count()

# Contagem antes da reexecução
total_antes = spark.table(TABELA_SILVER).count()

print("=== CONFERÊNCIA ANTES DO MERGE ===")
print(f"Registros no lote operacional: {total_lote:,}")
print(f"Registros na Silver antes: {total_antes:,}")

# Reexecuta o MERGE utilizando apenas a chave
df_lote_reexecucao.createOrReplaceTempView(
    "vw_lote_reexecucao_ids"
)

spark.sql(f"""
MERGE INTO {TABELA_SILVER} AS destino
USING vw_lote_reexecucao_ids AS origem
ON destino.Id = origem.Id
WHEN MATCHED THEN UPDATE SET destino.Id = origem.Id
""")

# Verifica as contagens após a reexecução
df_silver_validacao = spark.table(TABELA_SILVER)

total_depois = df_silver_validacao.count()
ids_distintos = df_silver_validacao.select("Id").distinct().count()
duplicatas = total_depois - ids_distintos

print("=== 1.3.2.a - TESTE DE IDEMPOTÊNCIA ===")
print(f"Registros antes: {total_antes:,}")
print(f"Registros depois: {total_depois:,}")
print(f"IDs duplicados excedentes: {duplicatas}")

if total_antes == total_depois and duplicatas == 0:
    print("VALIDAÇÃO APROVADA: reexecução sem duplicações.")
else:
    print("VALIDAÇÃO NÃO APROVADA.")


# COMMAND ----------

# DBTITLE 1,Reexecução

from pyspark.sql import functions as F

# 1.3.2.a - Reexecução da estratégia original de inserção

TABELA_SILVER = "workspace.sabor_dev.silver_reviews"

# Recupera os registros operacionais já normalizados
# pela transformação aplicada anteriormente
df_lote_completo = (
    df_silver_preparado
    .filter(F.col("recorte_temporal") == "operacional_1")
    .select(*spark.table(TABELA_SILVER).columns)
)

# Disponibiliza o lote completo para o MERGE
df_lote_completo.createOrReplaceTempView(
    "vw_lote_operacional_reprocessamento"
)

total_antes = spark.table(TABELA_SILVER).count()

# Reexecuta a mesma lógica da carga incremental
spark.sql(f"""
MERGE INTO {TABELA_SILVER} AS destino
USING vw_lote_operacional_reprocessamento AS origem
ON destino.Id = origem.Id
WHEN NOT MATCHED THEN INSERT *
""")

total_depois = spark.table(TABELA_SILVER).count()

duplicatas = (
    total_depois
    - spark.table(TABELA_SILVER).select("Id").distinct().count()
)

print("=== REEXECUÇÃO DO MERGE ORIGINAL ===")
print(f"Registros antes: {total_antes:,}")
print(f"Registros depois: {total_depois:,}")
print(f"Duplicatas excedentes: {duplicatas}")

if total_antes == total_depois and duplicatas == 0:
    print("IDEMPOTÊNCIA CONFIRMADA.")
else:
    print("VERIFICAR: reprocessamento apresentou diferença.")
