# Databricks notebook source
# MAGIC %pip install "mlflow-skinny[databricks]==3.12.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./05_configuracao_pipeline

# COMMAND ----------


import time
import mlflow
import mlflow.sklearn

from datetime import datetime, timezone
from uuid import uuid4
from mlflow import MlflowClient
from pyspark.sql import functions as F
from sklearn.metrics import precision_score, recall_score, f1_score

# =====================================================
# 1. CONFIGURACAO
# =====================================================

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

client = MlflowClient()

defaults = {
    "etapa": "treinar_carregar",
    "modo": "inferencia",
    "lote_id": "operacional_2",
    "ambiente": "dev",
    "job_run_id": "teste_manual_003"
}

for nome, valor in defaults.items():
    try:
        dbutils.widgets.get(nome)
    except Exception:
        dbutils.widgets.text(nome, valor)

p = {
    nome: dbutils.widgets.get(nome).strip()
    for nome in defaults
}

etapa = p["etapa"]
modo = p["modo"]
lote = p["lote_id"]
ambiente = p["ambiente"]
job_run_id = p["job_run_id"]

MODELO = "workspace.sabor_dev.modelo_baixa_nota"
PREPARADA = "workspace.sabor_operacao.dados_pipeline"
PREDICOES = "workspace.sabor_operacao.predicoes_reviews"
CONTEXTO = "workspace.sabor_operacao.modelo_pipeline"
METRICAS = "workspace.sabor_operacao.metricas_pipeline"
LOG = "workspace.sabor_operacao.logs_execucao"

tentativa = str(uuid4())
inicio = datetime.now(timezone.utc)
relogio = time.perf_counter()

entrada = 0
saida = 0
versao_dados = None


# =====================================================
# 2. LOG OPERACIONAL
# =====================================================

def registrar_log(status, erro=None):

    registro = [(
        tentativa,
        job_run_id,
        lote,
        ambiente,
        modo,
        etapa,
        inicio,
        datetime.now(timezone.utc),
        round(time.perf_counter() - relogio, 3),
        status,
        entrada,
        saida,
        0,
        versao_dados,
        erro
    )]

    schema = """
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

    spark.createDataFrame(
        registro, schema
    ).write.mode("append").saveAsTable(LOG)


# =====================================================
# 3. FUNCOES AUXILIARES
# =====================================================

def exigir_sucesso(tarefa):

    ultimo = (
        spark.table(LOG)
        .filter(
            (F.col("job_run_id") == job_run_id)
            & (F.col("lote_id") == lote)
            & (F.col("modo") == modo)
            & (F.col("ambiente") == ambiente)
            & (F.col("etapa") == tarefa)
        )
        .orderBy(F.col("inicio").desc())
        .first()
    )

    if ultimo is None or ultimo["status"] != "SUCESSO":
        raise ValueError(
            f"Tarefa {tarefa} não concluída com sucesso."
        )


def obter_dados():

    exigir_sucesso("validar")

    df = spark.table(PREPARADA).filter(
        (F.col("job_run_id") == job_run_id)
        & (F.col("lote_execucao") == lote)
        & (F.col("modo_execucao") == modo)
        & (F.col("ambiente_execucao") == ambiente)
    )

    if df.count() == 0:
        raise ValueError("Dados validados não encontrados.")

    return df


def obter_contexto_modelo():

    contexto = (
        spark.table(CONTEXTO)
        .filter(F.col("job_run_id") == job_run_id)
        .first()
    )

    if contexto is None:
        raise ValueError("Modelo da execução não identificado.")

    return contexto


# =====================================================
# 4. EXECUCAO DAS TAREFAS
# =====================================================

try:

    dados = obter_dados()

    versao_dados = int(
        dados.select("versao_fonte").first()[0]
    )

    print(f"Etapa: {etapa}")
    print(f"Modo: {modo}")
    print(f"Lote: {lote}")
    print(f"Job run: {job_run_id}")


    # =================================================
    # TAREFA 3 - TREINAR OU CARREGAR
    # =================================================

    if etapa == "treinar_carregar":

        entrada = dados.count()

        # Reutiliza os modelos treinados na Fase 3

        if modo == "treinamento":
            versao = client.get_model_version(MODELO, "2")

        elif modo == "retreinamento":
            versao = client.get_model_version(MODELO, "3")

        elif modo == "inferencia":
            versao = client.get_model_version_by_alias(
                MODELO, "champion"
            )

        else:
            raise ValueError("Modo inválido.")

        modelo = mlflow.sklearn.load_model(
            f"models:/{MODELO}/{versao.version}"
        )

        limiar = float(
            versao.tags["limiar_classificacao"]
        )

        contexto = [{
            "job_run_id": job_run_id,
            "modelo": MODELO,
            "versao_modelo": str(versao.version),
            "run_id": versao.run_id,
            "limiar": limiar
        }]

        (
            spark.createDataFrame(contexto)
            .write.format("delta")
            .mode("overwrite")
            .saveAsTable(CONTEXTO)
        )

        saida = 1

        print(f"Modelo carregado: {MODELO}")
        print(f"Versão: {versao.version}")
        print(f"Limiar: {limiar}")
        print("Modelo recuperado sem novo treinamento.")


    # =================================================
    # TAREFA 4 - INFERENCIA
    # =================================================

    elif etapa == "inferir":

        exigir_sucesso("treinar_carregar")

        ctx = obter_contexto_modelo()

        modelo = mlflow.sklearn.load_model(
            f"models:/{ctx['modelo']}/{ctx['versao_modelo']}"
        )

        df_lote = (
            dados.filter(F.col("recorte_temporal") == lote)
            .select("Id", "texto_entrada")
            .orderBy(F.xxhash64("Id", F.lit(42)), "Id")
            .limit(1000)
        )

        entrada = df_lote.count()

        if entrada == 0:
            raise ValueError("Lote de inferência vazio.")

        pdf = df_lote.toPandas()

        pdf["probabilidade_classe_1"] = (
            modelo.predict_proba(pdf["texto_entrada"])[:, 1]
        )

        pdf["classe_predita"] = (
            pdf["probabilidade_classe_1"] >= ctx["limiar"]
        ).astype(int)

        pdf["lote_id"] = lote
        pdf["modelo"] = ctx["modelo"]
        pdf["versao_modelo"] = ctx["versao_modelo"]
        pdf["run_id"] = ctx["run_id"]

        df_pred = (
            spark.createDataFrame(
                pdf.drop(columns=["texto_entrada"])
            )
            .withColumn(
                "instante_predicao",
                F.current_timestamp()
            )
        )

        df_pred.createOrReplaceTempView("vw_pred_job")

        spark.sql(f"""
        MERGE INTO {PREDICOES} AS d
        USING vw_pred_job AS o

        ON d.Id = o.Id
        AND d.lote_id = o.lote_id
        AND d.modelo = o.modelo
        AND d.versao_modelo = o.versao_modelo

        WHEN NOT MATCHED THEN INSERT *
        """)

        resultado = spark.table(PREDICOES).filter(
            (F.col("lote_id") == lote)
            & (F.col("modelo") == ctx["modelo"])
            & (
                F.col("versao_modelo")
                == ctx["versao_modelo"]
            )
        )

        saida = resultado.count()

        assert saida == entrada

        assert (
            resultado.select("Id").distinct().count()
            == saida
        )

        display(
            resultado.groupBy("classe_predita").count()
        )

        print(f"Predições: {saida}")
        print("Duplicidades: ZERO")


    # =================================================
    # TAREFA 5 - ATUALIZAR METRICAS
    # =================================================

    elif etapa == "atualizar_metricas":

        exigir_sucesso("inferir")

        ctx = obter_contexto_modelo()

        resultado = (
            spark.table(PREDICOES)
            .filter(
                (F.col("lote_id") == lote)
                & (F.col("modelo") == ctx["modelo"])
                & (
                    F.col("versao_modelo")
                    == ctx["versao_modelo"]
                )
            )
            .join(
                dados.select("Id", "alvo_baixa_nota"),
                "Id"
            )
        )

        entrada = resultado.count()

        if entrada == 0:
            raise ValueError("Predições não encontradas.")

        pdf = resultado.toPandas()

        y = pdf["alvo_baixa_nota"]
        pred = pdf["classe_predita"]

        metricas = [{
            "job_run_id": job_run_id,
            "lote_id": lote,
            "modelo": ctx["modelo"],
            "versao_modelo": ctx["versao_modelo"],
            "run_id": ctx["run_id"],
            "quantidade": entrada,
            "precision": float(
                precision_score(y, pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y, pred, zero_division=0)
            ),
            "f1": float(
                f1_score(y, pred, zero_division=0)
            )
        }]

        df_metricas = spark.createDataFrame(
            metricas
        ).withColumn(
            "instante_atualizacao",
            F.current_timestamp()
        )

        if not spark.catalog.tableExists(METRICAS):
            df_metricas.limit(0).write.format(
                "delta"
            ).saveAsTable(METRICAS)

        df_metricas.createOrReplaceTempView(
            "vw_metricas_pipeline"
        )

        spark.sql(f"""
        MERGE INTO {METRICAS} AS d
        USING vw_metricas_pipeline AS o

        ON d.lote_id = o.lote_id
        AND d.modelo = o.modelo
        AND d.versao_modelo = o.versao_modelo

        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """)

        saida = 1

        display(df_metricas)

        print("Métricas atualizadas com sucesso.")


    else:
        raise ValueError("Etapa inválida.")


    # =================================================
    # 5. FINALIZACAO
    # =================================================

    registrar_log("SUCESSO")

    print("\n=== RESULTADO FINAL ===")
    print(f"Etapa: {etapa}")
    print("Status: SUCESSO")
    print(f"Entrada: {entrada}")
    print(f"Saída: {saida}")
    print(f"Tentativa: {tentativa}")


except Exception as erro:

    registrar_log("FALHA", str(erro)[:4000])

    print(f"Falha na etapa {etapa}: {erro}")

    raise


display(
    spark.table(LOG)
    .filter(F.col("tentativa_id") == tentativa)
)
