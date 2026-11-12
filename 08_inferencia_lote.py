# Databricks notebook source
# MAGIC %run ./05_configuracao_pipeline

# COMMAND ----------

# MAGIC %pip install "mlflow-skinny[databricks]==3.12.0"

# COMMAND ----------

# MAGIC  %restart_python

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------


import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from pyspark.sql import functions as F

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

client = MlflowClient()

LOTE = "operacional_1"
MODELO = "workspace.sabor_dev.modelo_baixa_nota"
DESTINO = "workspace.sabor_operacao.predicoes_reviews"

# 1. Recupera o champion e o limiar registrado

versao = client.get_model_version_by_alias(
    MODELO, "champion"
)

limiar = float(
    versao.tags["limiar_classificacao"]
)

modelo = mlflow.sklearn.load_model(
    f"models:/{MODELO}/{versao.version}"
)

# 2. Seleciona 1.000 textos sem duplicidades

pdf = (
    spark.table(
        "workspace.sabor_dev.gold_reviews_ml_deduplicada"
    )
    .filter(F.col("recorte_temporal") == LOTE)
    .select("Id", "texto_entrada")
    .orderBy(
        F.xxhash64("Id", F.lit(42)),
        "Id"
    )
    .limit(1000)
    .toPandas()
)

# 3. Gera probabilidades e aplica o limiar

pdf["probabilidade_classe_1"] = (
    modelo.predict_proba(pdf["texto_entrada"])[:, 1]
)

pdf["classe_predita"] = (
    pdf["probabilidade_classe_1"] >= limiar
).astype(int)

pdf["lote_id"] = LOTE
pdf["versao_modelo"] = str(versao.version)
pdf["run_id"] = versao.run_id
pdf["modelo"] = MODELO

df_predicoes = (
    spark.createDataFrame(
        pdf.drop(columns=["texto_entrada"])
    )
    .withColumn(
        "instante_predicao",
        F.current_timestamp()
    )
)

# 4. Cria a tabela, caso ainda não exista

if not spark.catalog.tableExists(DESTINO):
    (
        df_predicoes.limit(0).write
        .format("delta")
        .saveAsTable(DESTINO)
    )

df_predicoes.createOrReplaceTempView("vw_predicoes")

# 5. Grava e reexecuta para testar idempotência

sql_merge = f"""
MERGE INTO {DESTINO} AS d
USING vw_predicoes AS o

ON d.Id = o.Id
AND d.lote_id = o.lote_id
AND d.modelo = o.modelo
AND d.versao_modelo = o.versao_modelo

WHEN NOT MATCHED THEN INSERT *
"""

spark.sql(sql_merge)

filtro = (
    (F.col("lote_id") == LOTE)
    & (F.col("modelo") == MODELO)
    & (F.col("versao_modelo") == str(versao.version))
)

antes = spark.table(DESTINO).filter(filtro).count()

spark.sql(sql_merge)

resultado = spark.table(DESTINO).filter(filtro)
depois = resultado.count()

assert antes == depois == len(pdf)
assert resultado.select("Id").distinct().count() == depois

# 6. Evidência consolidada

print("=== INFERÊNCIA OPERACIONAL ===")
print(f"Lote: {LOTE}")
print(f"Modelo: {MODELO}")
print(f"Versão champion: {versao.version}")
print(f"Run ID: {versao.run_id}")
print(f"Limiar: {limiar}")
print(f"Predições: {len(pdf)}")
print(f"Registros antes da reexecução: {antes}")
print(f"Registros depois da reexecução: {depois}")
print("Duplicidades: ZERO")
print("Retreinamento: NÃO REALIZADO")

display(resultado.groupBy("classe_predita").count())

display(resultado.limit(5))

print("INFERÊNCIA E IDEMPOTÊNCIA APROVADAS.")


# COMMAND ----------

# DBTITLE 1,3.4 - Monitoramento, retreinamento e reversão

import mlflow
import mlflow.sklearn
import pandas as pd

from pyspark.sql import functions as F
from sklearn.metrics import precision_score, recall_score, f1_score

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

MODELO = "workspace.sabor_dev.modelo_baixa_nota"
VERSAO = "2"
LOTE = "operacional_1"
RUN_ID = "3aa58e9aa92243e9b7cbc7a323356199"

# 1. Recupera os dados e as amostras originais

base = spark.table(
    "workspace.sabor_dev.gold_reviews_ml_deduplicada"
)

bronze = spark.table(
    "workspace.sabor_dev.bronze_reviews"
)

ids = mlflow.artifacts.load_dict(
    f"runs:/{RUN_ID}/ids_amostras.json"
)

df_treino = base.filter(
    F.col("Id").isin([int(i) for i in ids["treino"]])
).select("Id", "texto_entrada")

assert df_treino.count() == 7000

treino = df_treino.toPandas()

# 2. Recupera as predições já gravadas

predicoes = spark.table(
    "workspace.sabor_operacao.predicoes_reviews"
).filter(
    (F.col("lote_id") == LOTE)
    & (F.col("modelo") == MODELO)
    & (F.col("versao_modelo") == VERSAO)
)

assert predicoes.count() == 1000

# Somente agora recupera as notas reais

notas = bronze.select(
    F.col("Id").cast("long").alias("Id"),
    F.col("Score").cast("int").alias("Score")
)

operacional = (
    predicoes
    .join(base.select("Id", "texto_entrada"), "Id")
    .join(notas, "Id")
    .toPandas()
)

assert len(operacional) == 1000

operacional["classe_real"] = (
    operacional["Score"] <= 2
).astype(int)

# 3. Predições de referência no treino

modelo = mlflow.sklearn.load_model(
    f"models:/{MODELO}/{VERSAO}"
)

treino["classe_predita"] = modelo.predict(
    treino["texto_entrada"]
)

# 4. Indicador de mudança textual

media_treino = treino["texto_entrada"].str.len().mean()
media_lote = operacional["texto_entrada"].str.len().mean()

drift = (
    (media_lote - media_treino) / media_treino * 100
    if media_treino != 0 else None
)

alerta = (
    "ALERTA" if abs(drift) > 30 else "NORMAL"
) if drift is not None else "INDEFINIDO"

# 5. Qualidade dos dados nos períodos de origem

gold_original = spark.table(
    "workspace.sabor_dev.gold_reviews_ml"
).select("Id").withColumn("valido", F.lit(1))

origem = bronze.select(
    F.col("Id").cast("long").alias("Id"),
    F.col("Time").cast("long").alias("time_unix"),
    F.col("Text")
).join(gold_original, "Id", "left")

def taxa_qualidade(df):
    return df.agg(
        (
            F.avg(
                F.when(
                    F.col("Text").isNull()
                    | (F.trim("Text") == ""), 1
                ).otherwise(0)
            ) * 100
        ).alias("vazios"),
        (
            F.avg(
                F.when(F.col("valido").isNull(), 1)
                .otherwise(0)
            ) * 100
        ).alias("rejeitados")
    ).first()

q_treino = taxa_qualidade(
    origem.filter(F.col("time_unix") <= 1317427200)
)

q_lote = taxa_qualidade(
    origem.filter(
        (F.col("time_unix") > 1336521600)
        & (F.col("time_unix") <= 1344384000)
    )
)

# 6. Consolida os indicadores

metricas = {
    "lote_id": LOTE,
    "modelo": MODELO,
    "versao_modelo": VERSAO,
    "n_treino": len(treino),
    "n_operacional": len(operacional),
    "media_caracteres_treino": float(media_treino),
    "media_caracteres_lote": float(media_lote),
    "drift_texto_pct": float(drift) if drift is not None else None,
    "alerta_drift": alerta,
    "vazios_treino_pct": float(q_treino["vazios"]),
    "vazios_lote_pct": float(q_lote["vazios"]),
    "rejeitados_treino_pct": float(q_treino["rejeitados"]),
    "rejeitados_lote_pct": float(q_lote["rejeitados"]),
    "predicao_classe_1_treino_pct": float(
        treino["classe_predita"].mean() * 100
    ),
    "predicao_classe_1_lote_pct": float(
        operacional["classe_predita"].mean() * 100
    ),
    "precision_classe_1": float(precision_score(
        operacional["classe_real"],
        operacional["classe_predita"],
        zero_division=0
    )),
    "recall_classe_1": float(recall_score(
        operacional["classe_real"],
        operacional["classe_predita"],
        zero_division=0
    )),
    "f1_classe_1": float(f1_score(
        operacional["classe_real"],
        operacional["classe_predita"],
        zero_division=0
    ))
}

# 7. Persiste o monitoramento

destino = "workspace.sabor_operacao.monitoramento_reviews"

df_monitoramento = spark.createDataFrame(
    [tuple(metricas.values())],
    """
    lote_id STRING, modelo STRING,
    versao_modelo STRING, n_treino LONG,
    n_operacional LONG,
    media_caracteres_treino DOUBLE,
    media_caracteres_lote DOUBLE,
    drift_texto_pct DOUBLE, alerta_drift STRING,
    vazios_treino_pct DOUBLE, vazios_lote_pct DOUBLE,
    rejeitados_treino_pct DOUBLE,
    rejeitados_lote_pct DOUBLE,
    predicao_classe_1_treino_pct DOUBLE,
    predicao_classe_1_lote_pct DOUBLE,
    precision_classe_1 DOUBLE,
    recall_classe_1 DOUBLE, f1_classe_1 DOUBLE
    """
).withColumn("instante_monitoramento", F.current_timestamp())

if not spark.catalog.tableExists(destino):
    df_monitoramento.limit(0).write.format("delta").saveAsTable(destino)

df_monitoramento.createOrReplaceTempView("vw_monitoramento")

spark.sql(f"""
MERGE INTO {destino} AS d
USING vw_monitoramento AS o
ON d.lote_id = o.lote_id
AND d.modelo = o.modelo
AND d.versao_modelo = o.versao_modelo

WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")

# 8. Evidência consolidada

print("=== MONITORAMENTO OPERACIONAL ===")

display(pd.DataFrame(
    metricas.items(),
    columns=["indicador", "resultado"]
).astype(str))

print(f"\nTabela: {destino}")
print("MONITORAMENTO CONCLUIDO.")


# COMMAND ----------

display(
    spark.table(
        "workspace.sabor_operacao.monitoramento_reviews"
    ).select(
        "precision_classe_1",
        "recall_classe_1",
        "f1_classe_1"
    )
)

# COMMAND ----------


import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from pyspark.sql import functions as F
from sklearn.metrics import precision_score, recall_score, f1_score

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

client = MlflowClient()

MODELO = "workspace.sabor_dev.modelo_baixa_nota"
LOTE = "operacional_2"
DESTINO = "workspace.sabor_operacao.predicoes_reviews"

# 1. Carrega o champion atualizado

champion = client.get_model_version_by_alias(MODELO, "champion")

modelo = mlflow.sklearn.load_model(
    f"models:/{MODELO}/{champion.version}"
)

limiar = float(champion.tags["limiar_classificacao"])

# 2. Seleciona o segundo lote operacional

base = spark.table(
    "workspace.sabor_dev.gold_reviews_ml_deduplicada"
)

amostra = (
    base.filter(F.col("recorte_temporal") == LOTE)
    .select("Id", "texto_entrada")
    .orderBy(F.xxhash64("Id", F.lit(42)), "Id")
    .limit(1000)
)

assert amostra.count() == 1000

pdf = amostra.toPandas()

# 3. Classifica e persiste as predições

pdf["probabilidade_classe_1"] = (
    modelo.predict_proba(pdf["texto_entrada"])[:, 1]
)

pdf["classe_predita"] = (
    pdf["probabilidade_classe_1"] >= limiar
).astype(int)

pdf["lote_id"] = LOTE
pdf["modelo"] = MODELO
pdf["versao_modelo"] = str(champion.version)
pdf["run_id"] = champion.run_id

df_pred = spark.createDataFrame(
    pdf.drop(columns=["texto_entrada"])
).withColumn("instante_predicao", F.current_timestamp())

df_pred.createOrReplaceTempView("vw_predicoes_lote2")

spark.sql(f"""
MERGE INTO {DESTINO} AS d
USING vw_predicoes_lote2 AS o

ON d.Id = o.Id
AND d.lote_id = o.lote_id
AND d.modelo = o.modelo
AND d.versao_modelo = o.versao_modelo

WHEN NOT MATCHED THEN INSERT *
""")

# 4. Revela as notas e avalia o lote

resultado = (
    spark.table(DESTINO)
    .filter(
        (F.col("lote_id") == LOTE)
        & (F.col("modelo") == MODELO)
        & (F.col("versao_modelo") == str(champion.version))
    )
    .join(
        base.select("Id", "alvo_baixa_nota"),
        "Id"
    )
    .toPandas()
)

assert len(resultado) == 1000
assert resultado["Id"].nunique() == 1000

metricas = {
    "lote_id": LOTE,
    "modelo": MODELO,
    "versao_modelo": str(champion.version),
    "precision": float(precision_score(
        resultado["alvo_baixa_nota"],
        resultado["classe_predita"],
        zero_division=0
    )),
    "recall": float(recall_score(
        resultado["alvo_baixa_nota"],
        resultado["classe_predita"],
        zero_division=0
    )),
    "f1": float(f1_score(
        resultado["alvo_baixa_nota"],
        resultado["classe_predita"],
        zero_division=0
    ))
}

spark.createDataFrame([metricas]).write.mode(
    "overwrite"
).saveAsTable(
    "workspace.sabor_operacao.metricas_lote2"
)

print("=== AVALIAÇÃO DO SEGUNDO LOTE ===")
print(f"Versão: {champion.version}")
print(f"Predições: {len(resultado)}")

display(pd.DataFrame([metricas]))

# 5. Demonstra o rollback para a versão 2

versao_original = str(champion.version)

print("\n=== TESTE DE ROLLBACK ===")

try:
    client.set_registered_model_alias(
        MODELO, "champion", "2"
    )

    referencia = client.get_model_version_by_alias(
        MODELO, "champion"
    )

    modelo_anterior = mlflow.sklearn.load_model(
        f"models:/{MODELO}/{referencia.version}"
    )

    limiar_anterior = float(
        referencia.tags["limiar_classificacao"]
    )

    teste_rollback = pdf[["Id", "texto_entrada"]].head(10).copy()

    teste_rollback["probabilidade"] = (
        modelo_anterior.predict_proba(
            teste_rollback["texto_entrada"]
        )[:, 1]
    )

    teste_rollback["classe_predita"] = (
        teste_rollback["probabilidade"] >= limiar_anterior
    ).astype(int)

    print(f"Rollback para versão: {referencia.version}")

    display(teste_rollback.drop(columns=["texto_entrada"]))

finally:
    client.set_registered_model_alias(
        MODELO, "champion", versao_original
    )

restaurado = client.get_model_version_by_alias(
    MODELO, "champion"
)

assert str(restaurado.version) == versao_original

print(f"Champion restaurado: versão {restaurado.version}")
print("ROLLBACK E RESTAURAÇÃO CONCLUÍDOS.")


# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     lote_id,
# MAGIC     versao_modelo,
# MAGIC     COUNT(*) AS total_registros,
# MAGIC     COUNT(DISTINCT Id) AS ids_distintos,
# MAGIC     COUNT(*) - COUNT(DISTINCT Id) AS duplicidades
# MAGIC
# MAGIC FROM workspace.sabor_operacao.predicoes_reviews
# MAGIC
# MAGIC WHERE lote_id = 'operacional_2'
# MAGIC   AND versao_modelo = '3'
# MAGIC
# MAGIC GROUP BY lote_id, versao_modelo;