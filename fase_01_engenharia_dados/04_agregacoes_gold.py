# Databricks notebook source
# DBTITLE 1,Agregações, media

from pyspark.sql import functions as F

# ============================================================
# SABOR EM FOCO - QUESTÃO 1.4
# GOLD: AGREGAÇÕES E EVIDÊNCIAS DE QUALIDADE
# ============================================================

TABELA_SILVER = "workspace.sabor_dev.silver_reviews"
TABELA_GOLD_MENSAL = "workspace.sabor_dev.gold_avaliacoes_mensais"
TABELA_GOLD_PRODUTO = "workspace.sabor_dev.gold_avaliacoes_produto"

# ============================================================
# 1. LEITURA E VALIDAÇÃO DA SILVER
# ============================================================

print("=== 1. VALIDAÇÃO DA CAMADA SILVER ===")

df_silver = spark.table(TABELA_SILVER)

resumo_qualidade = df_silver.agg(
    F.count("*").alias("total"),
    F.countDistinct("Id").alias("ids_distintos"),
    F.sum(
        F.when(
            F.col("Id").isNull()
            | F.col("ProductId").isNull()
            | (F.trim("ProductId") == "")
            | F.col("Score").isNull()
            | (~F.col("Score").between(1, 5))
            | F.col("data_avaliacao").isNull()
            | F.col("alvo_baixa_nota").isNull()
            | (
                F.col("alvo_baixa_nota")
                != F.when(F.col("Score").isin(1, 2), 1)
                     .otherwise(0)
            ),
            1
        ).otherwise(0)
    ).alias("registros_invalidos")
).first()

total_silver = resumo_qualidade["total"]
duplicatas = total_silver - resumo_qualidade["ids_distintos"]
invalidos = resumo_qualidade["registros_invalidos"]

print(f"Total Silver: {total_silver:,}")
print(f"IDs duplicados excedentes: {duplicatas:,}")
print(f"Registros inválidos: {invalidos:,}")

if duplicatas != 0 or invalidos != 0:
    raise ValueError("Silver reprovada nas verificações de qualidade.")

print("QUALIDADE SILVER: APROVADA")

# ============================================================
# 2. CONSTRUÇÃO DAS AGREGAÇÕES GOLD
# ============================================================

print("\n=== 2. CONSTRUÇÃO DAS AGREGAÇÕES ===")

# Gold mensal
df_gold_mensal = (
    df_silver
    .withColumn(
        "mes_referencia",
        F.trunc("data_avaliacao", "month")
    )
    .groupBy("mes_referencia")
    .agg(
        F.count("*").alias("quantidade_avaliacoes"),
        F.round(F.avg("Score"), 2).alias("nota_media"),
        F.round(
            F.avg("alvo_baixa_nota"), 4
        ).alias("proporcao_notas_1_2")
    )
)

# Gold por produto
df_gold_produto = (
    df_silver
    .groupBy("ProductId")
    .agg(
        F.count("*").alias("quantidade_avaliacoes"),
        F.round(F.avg("Score"), 2).alias("nota_media"),
        F.round(
            F.avg("alvo_baixa_nota"), 4
        ).alias("proporcao_notas_1_2")
    )
)

print("Agregações mensal e por produto preparadas.")

# ============================================================
# 3. VALIDAÇÃO ANTES DA PERSISTÊNCIA
# ============================================================

print("\n=== 3. VALIDAÇÃO DAS AGREGAÇÕES ===")

total_mensal = (
    df_gold_mensal
    .agg(F.sum("quantidade_avaliacoes").alias("total"))
    .first()["total"]
)

total_produto = (
    df_gold_produto
    .agg(F.sum("quantidade_avaliacoes").alias("total"))
    .first()["total"]
)

assert total_silver == 511534, (
    "Volume inesperado na Silver. "
    "Revisar os lotes carregados antes de continuar."
)

assert total_mensal == total_silver, (
    "Divergência na agregação mensal."
)

assert total_produto == total_silver, (
    "Divergência na agregação por produto."
)

print(f"Total Silver: {total_silver:,}")
print(f"Total Gold mensal: {total_mensal:,}")
print(f"Total Gold produto: {total_produto:,}")
print("AGREGAÇÕES: APROVADAS")

# ============================================================
# 4. PERSISTÊNCIA DAS TABELAS DELTA
# ============================================================

print("\n=== 4. PERSISTÊNCIA DAS TABELAS GOLD ===")

# A sobrescrita fica restrita às duas tabelas Gold
# deste projeto. Não altera Bronze nem Silver.

(
    df_gold_mensal.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABELA_GOLD_MENSAL)
)

(
    df_gold_produto.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABELA_GOLD_PRODUTO)
)

print(f"Gold mensal: {TABELA_GOLD_MENSAL}")
print(f"Gold produto: {TABELA_GOLD_PRODUTO}")

# ============================================================
# 5. VALIDAÇÃO DAS TABELAS PERSISTIDAS
# ============================================================

print("\n=== 5. VALIDAÇÃO FINAL ===")

gold_mensal = spark.table(TABELA_GOLD_MENSAL)
gold_produto = spark.table(TABELA_GOLD_PRODUTO)

resumo_mensal = gold_mensal.agg(
    F.count("*").alias("grupos"),
    F.sum("quantidade_avaliacoes").alias("total")
).first()

resumo_produto = gold_produto.agg(
    F.count("*").alias("grupos"),
    F.sum("quantidade_avaliacoes").alias("total")
).first()

assert resumo_mensal["total"] == total_silver
assert resumo_produto["total"] == total_silver

assert gold_mensal.filter(
    F.col("mes_referencia").isNull()
    | F.col("nota_media").isNull()
    | F.col("proporcao_notas_1_2").isNull()
    | (~F.col("nota_media").between(1, 5))
    | (~F.col("proporcao_notas_1_2").between(0, 1))
).count() == 0

assert gold_produto.filter(
    F.col("ProductId").isNull()
    | F.col("nota_media").isNull()
    | F.col("proporcao_notas_1_2").isNull()
    | (~F.col("nota_media").between(1, 5))
    | (~F.col("proporcao_notas_1_2").between(0, 1))
).count() == 0

print(f"Meses armazenados: {resumo_mensal['grupos']:,}")
print(f"Produtos armazenados: {resumo_produto['grupos']:,}")
print(f"Avaliações Gold mensal: {resumo_mensal['total']:,}")
print(f"Avaliações Gold produto: {resumo_produto['total']:,}")

print("\nVALIDAÇÃO FINAL APROVADA")
print("Duas tabelas Gold persistidas em Delta.")
print("Sem perda de avaliações nas agregações.")

# ============================================================
# 6. EXIBIÇÃO DAS EVIDÊNCIAS
# ============================================================

print("\n=== GOLD MENSAL - ÚLTIMOS 12 MESES ===")

display(
    gold_mensal
    .orderBy(F.desc("mes_referencia"))
    .limit(12)
)

print("\n=== GOLD POR PRODUTO - TOP 20 ===")

display(
    gold_produto
    .orderBy(
        F.desc("quantidade_avaliacoes"),
        F.asc("ProductId")
    )
    .limit(20)
)

print("\n=== RESUMO FINAL ===")
print("Questão 1.4 - Agregações Gold executadas.")
print("Tabelas Delta disponíveis no catálogo workspace.")


# COMMAND ----------

# DBTITLE 1,Ranking de produtos
from pyspark.sql import functions as F
# ============================================================
# QUESTÃO 1.4.1.a - RANKING DE PRODUTOS POR BAIXA NOTA

TABELA_SILVER = "workspace.sabor_dev.silver_reviews"
VOLUME_MINIMO = 20

# Leitura Silver
df_silver = spark.table(TABELA_SILVER)

# Calcula indicadores por produto
df_ranking_produto = (
    df_silver
    .groupBy("ProductId")
    .agg(
        F.count("*").alias("quantidade_avaliacoes"),
        F.round(F.avg("Score"), 2).alias("nota_media"),
        F.avg(
            F.col("alvo_baixa_nota").cast("double")
        ).alias("proporcao_notas_1_2")
    )
    # Aplica o volume mínimo exigido
    .filter(
        F.col("quantidade_avaliacoes") >= VOLUME_MINIMO
    )
    # Converte a proporção para percentual de apresentação
    .withColumn(
        "taxa_baixa_nota_pct",
        F.round(F.col("proporcao_notas_1_2") * 100, 2)
    )
    # Ordena pela maior taxa de baixa nota
    .orderBy(
        F.desc("proporcao_notas_1_2"),
        F.desc("quantidade_avaliacoes"),
        F.asc("ProductId")
    )
)

# Quantidade de produtos elegíveis
total_produtos_elegiveis = df_ranking_produto.count()

print("=== 1.4.1.a - RANKING DE PRODUTOS ===")
print(f"Volume mínimo de avaliações: {VOLUME_MINIMO}")
print(f"Produtos elegíveis: {total_produtos_elegiveis:,}")

# Apresenta os 20 produtos com maior taxa de baixa nota
display(
    df_ranking_produto.select(
        "ProductId",
        "quantidade_avaliacoes",
        "nota_media",
        "taxa_baixa_nota_pct"
    ).limit(20)
)

# Validação do volume mínimo
menor_volume = (
    df_ranking_produto
    .agg(F.min("quantidade_avaliacoes").alias("minimo"))
    .first()["minimo"]
)

print("\n=== VALIDAÇÃO ===")
print(f"Menor volume entre produtos elegíveis: {menor_volume}")

if menor_volume >= VOLUME_MINIMO:
    print("VALIDAÇÃO APROVADA: todos os produtos possuem volume mínimo.")
else:
    raise ValueError("Existem produtos abaixo do volume mínimo.")


# COMMAND ----------

# DBTITLE 1,persistência da base para Machine Learning

from pyspark.sql import functions as F

# ============================================================
# QUESTÃO 1.4.1.b - TABELA PREPARADA PARA MACHINE LEARNING
# ============================================================

TABELA_BRONZE = "workspace.sabor_dev.bronze_reviews"
TABELA_ML = "workspace.sabor_dev.gold_reviews_ml"

# Cortes cronológicos validados na Evidência 07
CORTE_DESENVOLVIMENTO = 1336521600
CORTE_OPERACIONAL_1 = 1344384000

# Quantidades de referência dos três recortes
CONTAGENS_ESPERADAS = {
    "desenvolvimento": 454533,
    "operacional_1": 57001,
    "operacional_2": 56920
}
# ============================================================
# 1. LEITURA E NORMALIZAÇÃO
# ============================================================

df_bronze = spark.table(TABELA_BRONZE)

def limpar_texto(coluna):
    texto = F.coalesce(F.col(coluna), F.lit(""))

    texto = F.regexp_replace(
        texto, r"<[^>]*>", " "
    )

    texto = F.regexp_replace(
        texto, r"\s+", " "
    )

    return F.trim(texto)

df_ml_preparado = (
    df_bronze
    .select(
        F.expr("try_cast(Id AS BIGINT)").alias("Id"),
        F.col("ProductId"),
        F.expr("try_cast(Score AS INT)").alias("Score"),
        F.expr("try_cast(Time AS BIGINT)").alias("time_unix"),
        limpar_texto("Summary").alias("Summary_limpo"),
        limpar_texto("Text").alias("Text_limpo")
    )
    .withColumn(
        "data_avaliacao",
        F.to_date(F.timestamp_seconds("time_unix"))
    )
    .withColumn(
        "texto_entrada",
        F.trim(
            F.concat_ws(
                " ",
                F.col("Summary_limpo"),
                F.col("Text_limpo")
            )
        )
    )
    .withColumn(
        "alvo_baixa_nota",
        F.when(F.col("Score").isin(1, 2), F.lit(1))
         .when(F.col("Score").isin(3, 4, 5), F.lit(0))
    )
    .withColumn(
        "recorte_temporal",
        F.when(
            F.col("time_unix") <= CORTE_DESENVOLVIMENTO,
            F.lit("desenvolvimento")
        )
        .when(
            F.col("time_unix") <= CORTE_OPERACIONAL_1,
            F.lit("operacional_1")
        )
        .otherwise(F.lit("operacional_2"))
    )
)

# ============================================================
# 2. SELEÇÃO DAS COLUNAS PARA ML
# ============================================================

# Score serve apenas para gerar o alvo.
# UserId e ProfileName não entram na tabela de ML.

df_ml = df_ml_preparado.select(
    "Id",
    "ProductId",
    "time_unix",
    "data_avaliacao",
    "texto_entrada",
    "alvo_baixa_nota",
    "recorte_temporal"
)

# ============================================================
# 3. VALIDAÇÕES ANTES DA GRAVAÇÃO
# ============================================================

print("=== VALIDAÇÃO DA BASE DE ML ===")

total_bronze = df_bronze.count()
total_ml = df_ml.count()
ids_distintos = df_ml.select("Id").distinct().count()

invalidos = df_ml.filter(
    F.col("Id").isNull()
    | F.col("ProductId").isNull()
    | (F.trim("ProductId") == "")
    | F.col("time_unix").isNull()
    | F.col("data_avaliacao").isNull()
    | F.col("texto_entrada").isNull()
    | (F.trim("texto_entrada") == "")
    | F.col("alvo_baixa_nota").isNull()
    | F.col("recorte_temporal").isNull()
).count()

print(f"Total Bronze: {total_bronze:,}")
print(f"Total preparado para ML: {total_ml:,}")
print(f"IDs distintos: {ids_distintos:,}")
print(f"Registros inválidos: {invalidos:,}")

assert total_ml == total_bronze == 568454
assert ids_distintos == total_ml
assert invalidos == 0

# ============================================================
# 4. VALIDAÇÃO DOS RECORTES TEMPORAIS
# ============================================================

resumo_recortes = (
    df_ml
    .groupBy("recorte_temporal")
    .agg(
        F.count("*").alias("quantidade"),
        F.min("data_avaliacao").alias("data_inicial"),
        F.max("data_avaliacao").alias("data_final"),
        F.sum("alvo_baixa_nota").alias("notas_baixas")
    )
    .orderBy("data_inicial")
)

resultados = {
    linha["recorte_temporal"]: linha["quantidade"]
    for linha in resumo_recortes.collect()
}

assert resultados == CONTAGENS_ESPERADAS, (
    f"Recortes divergentes: {resultados}"
)

print("\n=== RECORTES TEMPORAIS ===")
display(resumo_recortes)

print("RECORTES TEMPORAIS: APROVADOS")

# ============================================================
# 5. PERSISTÊNCIA DA TABELA DELTA
# ============================================================

# Sobrescreve somente a tabela ML deste projeto.
# Não altera Bronze, Silver nem as Gold anteriores.

(
    df_ml.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABELA_ML)
)

# ============================================================
# 6. VALIDAÇÃO DA TABELA PERSISTIDA
# ============================================================

df_ml_persistido = spark.table(TABELA_ML)

total_persistido = df_ml_persistido.count()

assert total_persistido == 568454

print("\n=== RESULTADO FINAL ===")
print(f"Tabela: {TABELA_ML}")
print(f"Registros persistidos: {total_persistido:,}")
print("Formato: Delta")
print("Recortes: desenvolvimento, operacional_1 e operacional_2")
print("Coluna de entrada: texto_entrada")
print("Coluna alvo: alvo_baixa_nota")
print("VALIDAÇÃO FINAL APROVADA")


# COMMAND ----------

# DBTITLE 1,Reconciliação

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ============================================================
# QUESTÃO 1.4.2 - RECONCILIAÇÃO POR LOTE
# ============================================================

BRONZE = "workspace.sabor_dev.bronze_reviews"
SILVER = "workspace.sabor_dev.silver_reviews"
GOLD_ML = "workspace.sabor_dev.gold_reviews_ml"

CORTE_DESENVOLVIMENTO = 1336521600
CORTE_OPERACIONAL_1 = 1344384000

# ============================================================
# 1. CLASSIFICAÇÃO DOS REGISTROS DA BRONZE
# ============================================================

df_origem = (
    spark.table(BRONZE)
    .select(
        F.col("Id").alias("id_original"),
        F.col("ProductId"),
        F.col("Score"),
        F.col("Time"),
        F.col("Text")
    )
    .withColumn(
        "Id",
        F.expr("try_cast(id_original AS BIGINT)")
    )
    .withColumn(
        "time_unix",
        F.expr("try_cast(Time AS BIGINT)")
    )
    .withColumn(
        "data_avaliacao",
        F.to_date(F.timestamp_seconds("time_unix"))
    )
    .withColumn(
        "texto_limpo",
        F.trim(
            F.regexp_replace(
                F.regexp_replace(
                    F.coalesce(F.col("Text"), F.lit("")),
                    r"<[^>]*>",
                    " "
                ),
                r"\s+",
                " "
            )
        )
    )
    .withColumn(
        "recorte_temporal",
        F.when(
            F.col("time_unix").isNull(),
            F.lit("sem_recorte")
        )
        .when(
            F.col("time_unix") <= CORTE_DESENVOLVIMENTO,
            F.lit("desenvolvimento")
        )
        .when(
            F.col("time_unix") <= CORTE_OPERACIONAL_1,
            F.lit("operacional_1")
        )
        .otherwise("operacional_2")
    )
)

# ============================================================
# 2. REGRAS DE QUALIDADE E DUPLICIDADE
# ============================================================

# Regras aplicadas aos dados históricos
df_origem = df_origem.withColumn(
    "conteudo_valido",
    F.col("Id").isNotNull()
    & F.col("ProductId").isNotNull()
    & (F.trim("ProductId") != "")
    & F.coalesce(
        F.trim("Score").rlike(r"^[1-5]$"),
        F.lit(False)
    )
    & F.col("data_avaliacao").isNotNull()
    & (F.length("texto_limpo") > 0)
)

# Identifica repetições por Id em toda a Bronze
janela_id = (
    Window.partitionBy("Id")
    .orderBy(
        F.col("time_unix").asc_nulls_last(),
        F.col("ProductId").asc_nulls_last()
    )
)

df_classificado = (
    df_origem
    .withColumn(
        "ordem_id",
        F.row_number().over(janela_id)
    )
    .withColumn(
        "situacao",
        F.when(
            ~F.col("conteudo_valido"),
            "invalido"
        )
        .when(
            F.col("ordem_id") > 1,
            "duplicado"
        )
        .otherwise("valido")
    )
)

# As situações são exclusivas:
# uma linha inválida não é contada novamente
# como duplicada.

# ============================================================
# 3. RECONCILIAÇÃO DAS ENTRADAS
# ============================================================

df_reconciliacao = (
    df_classificado
    .groupBy("recorte_temporal")
    .agg(
        F.count("*").alias("entradas"),
        F.sum(
            F.when(F.col("situacao") == "invalido", 1)
             .otherwise(0)
        ).alias("invalidos"),
        F.sum(
            F.when(F.col("situacao") == "duplicado", 1)
             .otherwise(0)
        ).alias("duplicados"),
        F.sum(
            F.when(F.col("situacao") == "valido", 1)
             .otherwise(0)
        ).alias("saidas_validas")
    )
)

# ============================================================
# 4. CONFERÊNCIA COM AS TABELAS PERSISTIDAS
# ============================================================

df_gold_contagem = (
    spark.table(GOLD_ML)
    .groupBy("recorte_temporal")
    .count()
    .withColumnRenamed("count", "registros_gold_ml")
)

df_silver_contagem = (
    spark.table(SILVER)
    .groupBy("recorte_temporal")
    .count()
    .withColumnRenamed("count", "registros_silver")
)

df_resultado = (
    df_reconciliacao
    .join(
        df_gold_contagem,
        on="recorte_temporal",
        how="left"
    )
    .join(
        df_silver_contagem,
        on="recorte_temporal",
        how="left"
    )
    .fillna(
        0,
        subset=["registros_gold_ml", "registros_silver"]
    )
    .withColumn(
        "pendentes_silver",
        F.col("saidas_validas") - F.col("registros_silver")
    )
    .orderBy("recorte_temporal")
)

print("=== RECONCILIAÇÃO POR LOTE ===")
display(df_resultado)

# ============================================================
# 5. VALIDAÇÃO FINAL
# ============================================================

linhas = df_resultado.collect()

assert len(linhas) == 3, (
    "Quantidade inesperada de recortes temporais."
)

for linha in linhas:

    entradas = linha["entradas"]
    invalidos = linha["invalidos"]
    duplicados = linha["duplicados"]
    validos = linha["saidas_validas"]
    gold_ml = linha["registros_gold_ml"]
    silver = linha["registros_silver"]

    assert entradas == invalidos + duplicados + validos, (
        f"Reconciliação incorreta: {linha['recorte_temporal']}"
    )

    assert validos == gold_ml, (
        f"Divergência na Gold ML: {linha['recorte_temporal']}"
    )

    assert silver <= validos, (
        f"Silver excede os registros válidos: "
        f"{linha['recorte_temporal']}"
    )

total_entradas = sum(l["entradas"] for l in linhas)
total_invalidos = sum(l["invalidos"] for l in linhas)
total_duplicados = sum(l["duplicados"] for l in linhas)
total_validos = sum(l["saidas_validas"] for l in linhas)
total_gold = sum(l["registros_gold_ml"] for l in linhas)
total_silver = sum(l["registros_silver"] for l in linhas)

assert total_entradas == 568454
assert total_gold == total_validos

print("\n=== RESUMO GERAL ===")
print(f"Entradas históricas: {total_entradas:,}")
print(f"Inválidos: {total_invalidos:,}")
print(f"Duplicados excedentes: {total_duplicados:,}")
print(f"Saídas válidas: {total_validos:,}")
print(f"Registros Gold ML: {total_gold:,}")
print(f"Registros Silver: {total_silver:,}")
print(f"Pendentes na Silver: {total_validos - total_silver:,}")

print("\nRECONCILIAÇÃO APROVADA")
print("Entradas, rejeições e saídas conciliadas por lote.")
print("Sem dupla contagem entre inválidos e duplicados.")


# COMMAND ----------

# DBTITLE 1,Registro das versões Delta

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, TimestampType
)

# ============================================================
# 1.4.2.a - REGISTRO DAS VERSÕES DELTA
# ============================================================

TABELA_CONTROLE = "workspace.sabor_operacao.versoes_dados"
SNAPSHOT = "fase1_questao1_4_v1"

TABELAS = [
    "workspace.sabor_dev.bronze_reviews",
    "workspace.sabor_dev.silver_reviews",
    "workspace.sabor_dev.gold_avaliacoes_mensais",
    "workspace.sabor_dev.gold_avaliacoes_produto",
    "workspace.sabor_dev.gold_reviews_ml"
]

# 1. Consulta as versões atuais
registros = []

for tabela in TABELAS:
    historico = (
        spark.sql(f"DESCRIBE HISTORY {tabela}")
        .orderBy(F.desc("version"))
        .limit(1)
        .first()
    )

    registros.append((
        SNAPSHOT,
        tabela,
        int(historico["version"]),
        historico["timestamp"]
    ))

esquema = StructType([
    StructField("snapshot_id", StringType(), False),
    StructField("tabela", StringType(), False),
    StructField("versao_delta", LongType(), False),
    StructField("timestamp_versao", TimestampType(), True)
])

df_versoes = spark.createDataFrame(registros, esquema)

print("=== VERSÕES DELTA IDENTIFICADAS ===")
display(df_versoes)

# 2. Cria a tabela de controle, caso não exista
if not spark.catalog.tableExists(TABELA_CONTROLE):
    (
        df_versoes.limit(0).write
        .format("delta")
        .saveAsTable(TABELA_CONTROLE)
    )

# 3. Registra cada versão somente uma vez por snapshot
df_versoes.createOrReplaceTempView("vw_versoes_snapshot")

spark.sql(f"""
MERGE INTO {TABELA_CONTROLE} AS destino
USING vw_versoes_snapshot AS origem
ON destino.snapshot_id = origem.snapshot_id
AND destino.tabela = origem.tabela
WHEN NOT MATCHED THEN INSERT *
""")

# 4. Verifica as versões efetivamente registradas
df_registrado = (
    spark.table(TABELA_CONTROLE)
    .filter(F.col("snapshot_id") == SNAPSHOT)
)

total_registrado = df_registrado.count()
tabelas_distintas = df_registrado.select("tabela").distinct().count()

assert total_registrado == 5
assert tabelas_distintas == 5

print("\n=== CONTROLE DE VERSIONAMENTO ===")
display(
    df_registrado.orderBy("tabela")
)

print(f"Snapshot: {SNAPSHOT}")
print(f"Tabelas registradas: {total_registrado}")
print("VALIDAÇÃO APROVADA: versões Delta registradas.")
