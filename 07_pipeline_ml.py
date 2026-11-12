# Databricks notebook source
# DBTITLE 1,Preparando dependencias
# MAGIC %pip install "mlflow-skinny[databricks]"

# COMMAND ----------

# DBTITLE 1,importando config
# MAGIC %run ./05_configuracao_pipeline

# COMMAND ----------

# DBTITLE 1,Validação MLflow

import mlflow
import pandas as pd
import numpy as np
import sklearn

from mlflow import MlflowClient
from importlib.metadata import version

# ============================================================
# SABOR EM FOCO - AMBIENTE DE MACHINE LEARNING
# ============================================================

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

NOME_EXPERIMENTO = CONFIG["nome_experimento"]

experimento = mlflow.set_experiment(
    NOME_EXPERIMENTO
)

client = MlflowClient()

# Confirma acesso ao experimento
experimento_validado = client.get_experiment(
    experimento.experiment_id
)

assert experimento_validado is not None

print("=== DEPENDENCIAS DE MACHINE LEARNING ===")
print(f"MLflow: {mlflow.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")

print("\n=== CONFIGURACAO MLFLOW ===")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Registry URI: {mlflow.get_registry_uri()}")
print(f"Experimento: {experimento.name}")
print(f"Experiment ID: {experimento.experiment_id}")

print("\n=== MODELOS CONFIGURADOS ===")
print(f"Baseline: {MODELOS['baseline']}")
print(f"Candidato: {MODELOS['candidato']}")

print("\n=== FONTE VERSIONADA ===")
print(f"Tabela: {TABELAS['gold_ml']}")
print(f"Snapshot: {CONFIG['snapshot_dados']}")

print("\nVALIDACAO APROVADA")
print("Ambiente MLflow pronto para experimentos.")


# COMMAND ----------

# DBTITLE 1,EXPERIMENTACAO E SELECAO DE MODELOS

# ============================================================
# SABOR EM FOCO
# FASE 3 - EXPERIMENTACAO E SELECAO DE MODELOS
# Integracao com a Fase 2 - Orquestracao
# ============================================================

import json
import hashlib
import tempfile

import numpy as np
import pandas as pd
import mlflow

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. CONFIGURACAO
# ============================================================

SEMENTE = CONFIG["semente"]
LIMITE_AMOSTRA = CONFIG["amostra_treinamento"]
MAX_FEATURES = CONFIG["max_features_tfidf"]
RECALL_MINIMO = CONFIG["recall_minimo_classe_1"]

TABELA_ML = TABELAS["gold_ml"]
TABELA_VERSOES = TABELAS["versoes_dados"]

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(CONFIG["nome_experimento"])

assert 1 <= LIMITE_AMOSTRA <= 50000

# ============================================================
# 2. LEITURA DA VERSAO REGISTRADA
# ============================================================

versoes = (
    spark.table(TABELA_VERSOES)
    .filter(
        (
            F.col("snapshot_id")
            == CONFIG["snapshot_dados"]
        )
        & (
            F.col("tabela")
            == TABELA_ML
        )
    )
    .select("versao_delta")
    .collect()
)

assert len(versoes) == 1

VERSAO_DADOS = int(versoes[0]["versao_delta"])

df_dev = (
    spark.read
    .option("versionAsOf", VERSAO_DADOS)
    .table(TABELA_ML)
    .filter(
        F.col("recorte_temporal")
        == "desenvolvimento"
    )
    .select(
        "Id",
        "time_unix",
        "texto_entrada",
        "alvo_baixa_nota"
    )
)

total_desenvolvimento = df_dev.count()

print("=== FONTE DE TREINAMENTO ===")
print(f"Tabela: {TABELA_ML}")
print(f"Versao Delta: {VERSAO_DADOS}")
print(f"Registros desenvolvimento: {total_desenvolvimento:,}")

assert total_desenvolvimento == RECORTES["desenvolvimento"]

# ============================================================
# 3. REMOCAO DE REPETICOES TEXTUAIS
# ============================================================

# A assinatura considera texto normalizado.
# O texto original utilizado no modelo permanece preservado.

df_dev = df_dev.withColumn(
    "assinatura_textual",
    F.sha2(
        F.lower(
            F.trim(
                F.regexp_replace(
                    "texto_entrada",
                    r"\s+",
                    " "
                )
            )
        ),
        256
    )
)

# Mantem a primeira ocorrencia cronologica
# de cada assinatura textual.

janela_texto = (
    Window
    .partitionBy("assinatura_textual")
    .orderBy(
        F.asc("time_unix"),
        F.asc("Id")
    )
)

df_sem_repeticoes = (
    df_dev
    .withColumn(
        "ordem_texto",
        F.row_number().over(janela_texto)
    )
    .filter(F.col("ordem_texto") == 1)
    .drop("ordem_texto")
)

total_sem_repeticoes = df_sem_repeticoes.count()

print("\n=== CONTROLE DE REPETICOES ===")

print(
    "Repeticoes removidas:",
    f"{total_desenvolvimento - total_sem_repeticoes:,}"
)

print(
    "Registros restantes:",
    f"{total_sem_repeticoes:,}"
)

# ============================================================
# 4. DIVISAO TEMPORAL
# ============================================================

# Define os cortes pela distribuicao cronologica
# completa, antes da amostragem.

distribuicao_tempo = (
    df_sem_repeticoes
    .groupBy("time_unix")
    .count()
    .orderBy("time_unix")
    .toPandas()
)

distribuicao_tempo["acumulado"] = (
    distribuicao_tempo["count"].cumsum()
)

alvo_treino = (
    total_sem_repeticoes
    * CONFIG["proporcao_treino"]
)

alvo_validacao = (
    total_sem_repeticoes
    * (
        CONFIG["proporcao_treino"]
        + CONFIG["proporcao_validacao"]
    )
)

indice_treino = (
    distribuicao_tempo["acumulado"]
    .sub(alvo_treino)
    .abs()
    .idxmin()
)

indice_validacao = (
    distribuicao_tempo["acumulado"]
    .sub(alvo_validacao)
    .abs()
    .idxmin()
)

CORTE_TREINO = int(
    distribuicao_tempo.loc[
        indice_treino,
        "time_unix"
    ]
)

CORTE_VALIDACAO = int(
    distribuicao_tempo.loc[
        indice_validacao,
        "time_unix"
    ]
)

assert CORTE_TREINO < CORTE_VALIDACAO

df_dividido = (
    df_sem_repeticoes
    .withColumn(
        "conjunto_ml",
        F.when(
            F.col("time_unix") <= CORTE_TREINO,
            "treino"
        )
        .when(
            F.col("time_unix") <= CORTE_VALIDACAO,
            "validacao"
        )
        .otherwise("teste")
    )
)

print("\n=== DIVISAO TEMPORAL COMPLETA ===")

display(
    df_dividido
    .groupBy("conjunto_ml")
    .agg(
        F.count("*").alias("quantidade"),
        F.min("time_unix").alias("tempo_inicial"),
        F.max("time_unix").alias("tempo_final")
    )
    .orderBy("tempo_inicial")
)

# ============================================================
# 5. AMOSTRAGEM REPRODUZIVEL
# ============================================================

n_treino = int(
    LIMITE_AMOSTRA
    * CONFIG["proporcao_treino"]
)

n_validacao = int(
    LIMITE_AMOSTRA
    * CONFIG["proporcao_validacao"]
)

n_teste = (
    LIMITE_AMOSTRA
    - n_treino
    - n_validacao
)

def selecionar_amostra(conjunto, quantidade):

    return (
        df_dividido
        .filter(
            F.col("conjunto_ml") == conjunto
        )
        .withColumn(
            "ordem_amostra",
            F.xxhash64(
                F.col("Id"),
                F.lit(SEMENTE)
            )
        )
        .orderBy(
            "ordem_amostra",
            "Id"
        )
        .limit(quantidade)
        .drop("ordem_amostra")
        .toPandas()
    )

pdf_treino = selecionar_amostra(
    "treino", n_treino
)

pdf_validacao = selecionar_amostra(
    "validacao", n_validacao
)

pdf_teste = selecionar_amostra(
    "teste", n_teste
)

assert len(pdf_treino) == n_treino
assert len(pdf_validacao) == n_validacao
assert len(pdf_teste) == n_teste

# Confere que os textos nao se repetem
# entre os conjuntos.

hashes_treino = set(
    pdf_treino["assinatura_textual"]
)

hashes_validacao = set(
    pdf_validacao["assinatura_textual"]
)

hashes_teste = set(
    pdf_teste["assinatura_textual"]
)

assert not hashes_treino.intersection(
    hashes_validacao
)

assert not hashes_treino.intersection(
    hashes_teste
)

assert not hashes_validacao.intersection(
    hashes_teste
)

# Confere a ordem temporal

assert (
    pdf_treino["time_unix"].max()
    < pdf_validacao["time_unix"].min()
)

assert (
    pdf_validacao["time_unix"].max()
    < pdf_teste["time_unix"].min()
)

print("\n=== AMOSTRAS DE MACHINE LEARNING ===")

print(f"Treino: {len(pdf_treino):,}")
print(f"Validacao: {len(pdf_validacao):,}")
print(f"Teste: {len(pdf_teste):,}")

print("Separacao temporal: APROVADA")
print("Sobreposicao de assinaturas: ZERO")

# ============================================================
# 6. VARIAVEIS DE ENTRADA E ALVO
# ============================================================

# Somente texto_entrada e utilizado como preditor.

X_train = pdf_treino["texto_entrada"]
y_train = pdf_treino["alvo_baixa_nota"].astype(int)

X_val = pdf_validacao["texto_entrada"]
y_val = pdf_validacao["alvo_baixa_nota"].astype(int)

X_test = pdf_teste["texto_entrada"]
y_test = pdf_teste["alvo_baixa_nota"].astype(int)

for nome, alvo in [
    ("treino", y_train),
    ("validacao", y_val),
    ("teste", y_test)
]:
    assert set(alvo.unique()) == {0, 1}, (
        f"Conjunto {nome} nao contem as duas classes."
    )

# ============================================================
# 7. TF-IDF E MODELOS
# ============================================================

# O TF-IDF sera ajustado exclusivamente no treino.
# Nao removemos stop words, preservando negacoes.

def criar_pipeline(classificador):

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=MAX_FEATURES,
                ngram_range=(1, 2),
                min_df=2,
                lowercase=True,
                dtype=np.float32
            )
        ),
        ("classificador", classificador)
    ])

modelos = {

    "baseline_dummy": criar_pipeline(
        DummyClassifier(
            strategy="most_frequent"
        )
    ),

    "regressao_logistica": criar_pipeline(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=SEMENTE
        )
    ),

    "multinomial_nb": criar_pipeline(
        MultinomialNB(
            alpha=0.5
        )
    )
}

# ============================================================
# 8. FUNCAO DE METRICAS
# ============================================================

def calcular_metricas(y_real, y_pred):

    return {
        "accuracy": float(
            accuracy_score(y_real, y_pred)
        ),
        "precision_classe_1": float(
            precision_score(
                y_real,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        ),
        "recall_classe_1": float(
            recall_score(
                y_real,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        ),
        "f1_classe_1": float(
            f1_score(
                y_real,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        )
    }

# ============================================================
# 9. EXPERIMENTOS MLFLOW
# ============================================================

resultados = []
modelos_treinados = {}

for nome, pipeline in modelos.items():

    print(f"\n=== TREINANDO: {nome} ===")

    with mlflow.start_run(
        run_name=nome
    ) as run:

        pipeline.fit(
            X_train,
            y_train
        )

        pred_val = pipeline.predict(X_val)

        metricas = calcular_metricas(
            y_val,
            pred_val
        )

        mlflow.log_params({
            "modelo": nome,
            "semente": SEMENTE,
            "versao_dados": VERSAO_DADOS,
            "snapshot": CONFIG["snapshot_dados"],
            "tabela_origem": TABELA_ML,
            "max_features_tfidf": MAX_FEATURES,
            "ngram_range": "1,2",
            "min_df": 2,
            "n_treino": len(pdf_treino),
            "n_validacao": len(pdf_validacao),
            "n_teste": len(pdf_teste),
            "corte_treino": CORTE_TREINO,
            "corte_validacao": CORTE_VALIDACAO,
            "recorte": "desenvolvimento",
            "assinatura_textual": "sha256",
            "limiar_classificacao": 0.5
        })

        # Parametros especificos do classificador
        classificador = (
            pipeline.named_steps["classificador"]
        )

        mlflow.log_params({
            f"clf_{chave}": str(valor)
            for chave, valor in (
                classificador.get_params().items()
            )
        })

        mlflow.log_metrics({
            f"val_{chave}": valor
            for chave, valor in metricas.items()
        })

        matriz = confusion_matrix(
            y_val,
            pred_val,
            labels=[0, 1]
        )

        mlflow.log_dict(
            {
                "matriz_confusao": matriz.tolist(),
                "metricas": metricas
            },
            "validacao_resultados.json"
        )

        # Registra os IDs das amostras utilizadas.
        # Permite reproduzir os conjuntos exatos.

        ids_amostras = {
            "treino": [
                int(x)
                for x in pdf_treino["Id"]
            ],
            "validacao": [
                int(x)
                for x in pdf_validacao["Id"]
            ],
            "teste": [
                int(x)
                for x in pdf_teste["Id"]
            ]
        }

        mlflow.log_dict(
            ids_amostras,
            "ids_amostras.json"
        )

        run_id = run.info.run_id

        resultados.append({
            "modelo": nome,
            "run_id": run_id,
            **metricas
        })

        modelos_treinados[nome] = pipeline

        print(f"Run ID: {run_id}")
        print(metricas)

# ============================================================
# 10. SELECAO DO CANDIDATO
# ============================================================

df_resultados = pd.DataFrame(resultados)

baseline_f1 = float(
    df_resultados.loc[
        df_resultados["modelo"]
        == "baseline_dummy",
        "f1_classe_1"
    ].iloc[0]
)

df_resultados["elegivel"] = (
    (
        df_resultados["modelo"]
        != "baseline_dummy"
    )
    & (
        df_resultados["f1_classe_1"]
        > baseline_f1
    )
    & (
        df_resultados["recall_classe_1"]
        >= RECALL_MINIMO
    )
)

df_resultados = (
    df_resultados
    .sort_values(
        ["elegivel", "f1_classe_1"],
        ascending=[False, False]
    )
)

print("\n=== COMPARACAO DOS MODELOS ===")

display(df_resultados)

elegiveis = df_resultados[
    df_resultados["elegivel"]
]

if elegiveis.empty:

    raise RuntimeError(
        "Nenhum candidato atende aos criterios "
        "de promocao. Ajustar treinamento antes "
        "de registrar ou publicar o modelo."
    )

melhor = elegiveis.iloc[0]

NOME_CANDIDATO = melhor["modelo"]
RUN_CANDIDATO = melhor["run_id"]

modelo_candidato = modelos_treinados[
    NOME_CANDIDATO
]

print("\n=== CANDIDATO SELECIONADO ===")

print(f"Modelo: {NOME_CANDIDATO}")
print(f"Run ID: {RUN_CANDIDATO}")
print(f"F1 baseline: {baseline_f1:.4f}")

print(
    f"F1 candidato: "
    f"{melhor['f1_classe_1']:.4f}"
)

print(
    f"Recall candidato: "
    f"{melhor['recall_classe_1']:.4f}"
)

print("REGRA DE PROMOCAO: APROVADA")

# ============================================================
# 11. AVALIACAO FINAL NO TESTE
# ============================================================

# O teste nao participou da selecao.

pred_teste = modelo_candidato.predict(
    X_test
)

metricas_teste = calcular_metricas(
    y_test,
    pred_teste
)

matriz_teste = confusion_matrix(
    y_test,
    pred_teste,
    labels=[0, 1]
)

with mlflow.start_run(
    run_id=RUN_CANDIDATO
):

    mlflow.log_metrics({
        f"test_{chave}": valor
        for chave, valor in metricas_teste.items()
    })

    mlflow.log_dict(
        {
            "metricas": metricas_teste,
            "matriz_confusao": matriz_teste.tolist()
        },
        "teste_resultados.json"
    )

print("\n=== AVALIACAO FINAL - TESTE ===")

print(metricas_teste)

print("\nMatriz de confusao:")
print(matriz_teste)

print("\nRelatorio de classificacao:")

print(
    classification_report(
        y_test,
        pred_teste,
        labels=[0, 1],
        zero_division=0
    )
)

# ============================================================
# 12. ANALISE DE CINCO ERROS
# ============================================================

df_erros = pdf_teste[
    ["Id", "texto_entrada", "alvo_baixa_nota"]
].copy()

df_erros["classe_predita"] = pred_teste

df_erros = df_erros[
    df_erros["alvo_baixa_nota"]
    != df_erros["classe_predita"]
].head(5)

df_erros["texto_entrada"] = (
    df_erros["texto_entrada"]
    .str.slice(0, 500)
)

print("\n=== AMOSTRA DE CINCO ERROS ===")

display(df_erros)

# ============================================================
# 13. RESUMO FINAL
# ============================================================

print("\n=== RESUMO FINAL DOS EXPERIMENTOS ===")

print(f"Versao dos dados: {VERSAO_DADOS}")

print(
    f"Total desenvolvimento: "
    f"{total_desenvolvimento:,}"
)

print(
    f"Repeticoes removidas: "
    f"{total_desenvolvimento - total_sem_repeticoes:,}"
)

print(f"Total amostra: {LIMITE_AMOSTRA:,}")
print("Modelos avaliados: 3")
print(f"Modelo selecionado: {NOME_CANDIDATO}")
print(f"Run selecionado: {RUN_CANDIDATO}")
print("Separacao cronologica: APROVADA")
print("Sobreposicao textual: ZERO")
print("Regra de promocao: APROVADA")

print("\nEXPERIMENTACAO CONCLUIDA.")


# COMMAND ----------


import numpy as np
import mlflow.sklearn

from mlflow import MlflowClient
from mlflow.models import infer_signature

client = MlflowClient()

# Entrada explicitamente textual
exemplo = np.array(
    X_train.iloc[:3].astype(str).tolist(),
    dtype=str
)

registros = [
    ("baseline_dummy", MODELOS["baseline"], ["baseline"]),
    ("regressao_logistica", MODELOS["candidato"], ["candidate", "champion"])
]

for nome, destino, aliases in registros:

    modelo = modelos_treinados[nome]

    run_id = df_resultados.loc[
        df_resultados["modelo"] == nome, "run_id"
    ].iloc[0]

    # Assinatura explícita: textos -> classes previstas
    assinatura = infer_signature(
        exemplo,
        modelo.predict(exemplo)
    )

    with mlflow.start_run(run_id=run_id):

        info = mlflow.sklearn.log_model(
            sk_model=modelo,
            name="model",
            signature=assinatura,
            registered_model_name=destino
        )

    versao = info.registered_model_version

    client.set_model_version_tag(
        destino, versao,
        "limiar_classificacao", "0.5"
    )

    for alias in aliases:
        client.set_registered_model_alias(
            destino, alias, versao
        )

    print(f"\nModelo: {destino}")
    print(f"Versão: {versao}")
    print(f"Aliases: {aliases}")
    print(f"Assinatura: {assinatura}")

print("\nREGISTRO DOS MODELOS CONCLUÍDO.")


# COMMAND ----------


import mlflow
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# =====================================================
# 3.1 - PREPARAÇÃO E PREVENÇÃO DE VAZAMENTO
# =====================================================

mlflow.set_tracking_uri("databricks")

RUN_ID = "3aa58e9aa92243e9b7cbc7a323356199"

TABELA_ML = TABELAS["gold_ml"]
TABELA_LIMPA = f"{BASE_DEV}.gold_reviews_ml_deduplicada"

# 1. Recupera as amostras originais do MLflow

ids = mlflow.artifacts.load_dict(
    f"runs:/{RUN_ID}/ids_amostras.json"
)

df = (
    spark.read
    .option("versionAsOf", 0)
    .table(TABELA_ML)
)

registros = [
    (int(id_avaliacao), conjunto)
    for conjunto, valores in ids.items()
    for id_avaliacao in valores
]

df_ids = spark.createDataFrame(
    registros, ["Id", "conjunto_ml"]
)

df_amostras = df.join(df_ids, "Id")

print("=== DISTRIBUIÇÃO DAS AMOSTRAS ===")

display(
    df_amostras
    .groupBy("conjunto_ml", "alvo_baixa_nota")
    .agg(
        F.count("*").alias("quantidade"),
        F.min("data_avaliacao").alias("data_inicial"),
        F.max("data_avaliacao").alias("data_final")
    )
    .orderBy("conjunto_ml", "alvo_baixa_nota")
)

assert df_amostras.count() == 10000

# 2. Gera assinatura textual para todos os recortes

df = df.withColumn(
    "assinatura_textual",
    F.sha2(
        F.lower(
            F.trim(
                F.regexp_replace(
                    "texto_entrada", r"\s+", " "
                )
            )
        ),
        256
    )
)

# 3. Recupera as notas originais para analisar conflitos

df_notas = (
    spark.read
    .option("versionAsOf", 0)
    .table(TABELAS["bronze"])
    .select(
        F.col("Id").cast("long").alias("Id"),
        F.col("Score").cast("int").alias("Score")
    )
)

conflitos = (
    df.join(df_notas, "Id")
    .groupBy("assinatura_textual")
    .agg(
        F.count("*").alias("quantidade"),
        F.countDistinct("Score").alias("notas_distintas"),
        F.sort_array(
            F.collect_set("Score")
        ).alias("notas_encontradas")
    )
    .filter(F.col("notas_distintas") > 1)
)

print("=== TEXTOS IGUAIS COM NOTAS CONFLITANTES ===")
print(f"Assinaturas conflitantes: {conflitos.count():,}")

display(conflitos.limit(10))

# 4. Mantém a ocorrência mais antiga de cada texto

janela = (
    Window.partitionBy("assinatura_textual")
    .orderBy("time_unix", "Id")
)

df_limpo = (
    df.withColumn("ordem", F.row_number().over(janela))
    .filter(F.col("ordem") == 1)
    .drop("ordem")
)

# 5. Compara os recortes antes e depois

antes = (
    df.groupBy("recorte_temporal")
    .count()
    .withColumnRenamed("count", "antes")
)

depois = (
    df_limpo.groupBy("recorte_temporal")
    .count()
    .withColumnRenamed("count", "depois")
)

print("=== DEDUPLICAÇÃO POR RECORTE ===")

display(
    antes.join(depois, "recorte_temporal")
    .withColumn(
        "repeticoes_removidas",
        F.col("antes") - F.col("depois")
    )
)

# 6. Persiste e valida a base deduplicada

(
    df_limpo.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(TABELA_LIMPA)
)

df_final = spark.table(TABELA_LIMPA)

total = df_final.count()
assinaturas = df_final.select(
    "assinatura_textual"
).distinct().count()

assert total == assinaturas

# Confirma que nenhuma amostra original foi perdida
amostras_preservadas = (
    df_final.join(df_ids, "Id").count()
)

assert amostras_preservadas == 10000

print("=== RESULTADO FINAL ===")
print(f"Tabela: {TABELA_LIMPA}")
print(f"Registros preservados: {total:,}")
print(f"Assinaturas distintas: {assinaturas:,}")
print(f"Amostras originais preservadas: {amostras_preservadas:,}")
print("Duplicidades textuais: ZERO")
print("VALIDAÇÃO APROVADA")


# COMMAND ----------


import mlflow
import mlflow.sklearn
import sklearn
import pandas as pd
import numpy as np
import scipy
import sys

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import average_precision_score
from mlflow.models import infer_signature

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Runs originais dos experimentos
runs = {
    "baseline_dummy": "5113938390c24fc5bc12216af0de5a05",
    "regressao_logistica": "3aa58e9aa92243e9b7cbc7a323356199",
    "multinomial_nb": "f47dbfcf25af4e2884658c0f216798e2"
}

# 1. Recupera as amostras originais
ids = mlflow.artifacts.load_dict(
    f"runs:/{runs['regressao_logistica']}/ids_amostras.json"
)

base = spark.table(
    "workspace.sabor_dev.gold_reviews_ml_deduplicada"
)

amostras = {}

for conjunto in ["treino", "validacao"]:
    identificadores = [int(i) for i in ids[conjunto]]

    pdf = (
        base.filter(base.Id.isin(identificadores))
        .select("Id", "texto_entrada", "alvo_baixa_nota")
        .toPandas()
        .set_index("Id")
        .loc[identificadores]
    )

    amostras[conjunto] = pdf

X_train = amostras["treino"]["texto_entrada"]
y_train = amostras["treino"]["alvo_baixa_nota"]

X_val = amostras["validacao"]["texto_entrada"]
y_val = amostras["validacao"]["alvo_baixa_nota"]

# 2. Recupera os modelos registrados
modelos = {
    "baseline_dummy": mlflow.sklearn.load_model(
        "models:/workspace.sabor_dev.modelo_baseline/1"
    ),
    "regressao_logistica": mlflow.sklearn.load_model(
        "models:/workspace.sabor_dev.modelo_baixa_nota/1"
    )
}

# Reconstrói somente o Naive Bayes original
modelos["multinomial_nb"] = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        dtype=np.float32
    )),
    ("clf", MultinomialNB(alpha=0.5))
])

modelos["multinomial_nb"].fit(X_train, y_train)

# 3. Complementa os registros do MLflow
resultados = []

for nome, modelo in modelos.items():

    prob = modelo.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, prob)

    with mlflow.start_run(run_id=runs[nome]):

        mlflow.log_metric("val_average_precision", ap)

        mlflow.log_params({
            "python_version": sys.version.split()[0],
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "mlflow_version": mlflow.__version__
        })

        mlflow.log_dict({
            "treino": {"0": 6038, "1": 962},
            "validacao": {"0": 1254, "1": 246},
            "teste": {"0": 1203, "1": 297}
        }, "distribuicao_classes.json")

        # Persiste o Pipeline que estava faltando
        if nome == "multinomial_nb":

            exemplo = X_train.iloc[:3].to_numpy(dtype=str)

            mlflow.sklearn.log_model(
                sk_model=modelo,
                name="model",
                signature=infer_signature(
                    exemplo, modelo.predict(exemplo)
                )
            )

    resultados.append({
        "modelo": nome,
        "average_precision": ap,
        "run_id": runs[nome]
    })

# 4. Resultado consolidado
print("=== AVERAGE PRECISION - VALIDACAO ===")

display(pd.DataFrame(resultados))

print("Distribuicao das classes registrada.")
print("Versoes das bibliotecas registradas.")
print("Pipeline Naive Bayes persistido.")
print("COMPLEMENTACAO MLFLOW CONCLUIDA.")


# COMMAND ----------

# DBTITLE 1,questão 3.4

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import numpy as np

from mlflow import MlflowClient
from mlflow.models import infer_signature

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

client = MlflowClient()

# 1. Complementa somente o candidato

destino = "workspace.sabor_dev.modelo_baixa_nota"

anterior = client.get_model_version(destino, "1")

modelo = mlflow.sklearn.load_model(
    f"models:/{destino}/1"
)

exemplo = np.array([
    "Excellent product, very tasty.",
    "Poor quality, disappointing taste."
])

with mlflow.start_run(run_id=anterior.run_id):

    info = mlflow.sklearn.log_model(
        sk_model=modelo,
        name="modelo_completo",
        signature=infer_signature(
            exemplo, modelo.predict(exemplo)
        ),
        input_example=exemplo,
        registered_model_name=destino
    )

versao = info.registered_model_version

client.set_model_version_tag(
    destino, versao,
    "limiar_classificacao", "0.5"
)

for alias in ["candidate", "champion"]:
    client.set_registered_model_alias(
        destino, alias, versao
    )

# 2. Confere os dois modelos registrados

for nome, v in [
    ("modelo_baseline", "2"),
    ("modelo_baixa_nota", versao)
]:

    uri = f"models:/workspace.sabor_dev.{nome}/{v}"

    print(f"\nModelo: {nome}")
    print(f"Versão: {v}")

    metadados = mlflow.models.get_model_info(uri)

    print("Assinatura:", metadados.signature)
    print("Exemplo:", metadados.saved_input_example_info)

    caminho = mlflow.pyfunc.get_model_dependencies(uri)

    print("Dependências:")

    with open(caminho) as arquivo:
        print(arquivo.read())

print("\nREGISTRO COMPLETO.")


# COMMAND ----------

# DBTITLE 1,3.4 monitoramento

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import functions as F
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score
)

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Shared/sabor_em_foco")

client = MlflowClient()

MODELO = "workspace.sabor_dev.modelo_baixa_nota"

# 1. Recupera o modelo atual e os dados

atual = client.get_model_version_by_alias(MODELO, "champion")

modelo_atual = mlflow.sklearn.load_model(
    f"models:/{MODELO}/{atual.version}"
)

base = spark.table(
    "workspace.sabor_dev.gold_reviews_ml_deduplicada"
)

ids = mlflow.artifacts.load_dict(
    "runs:/3aa58e9aa92243e9b7cbc7a323356199/ids_amostras.json"
)

historico = base.filter(
    F.col("Id").isin([int(i) for i in ids["treino"]])
)

lote = base.filter(
    F.col("recorte_temporal") == "operacional_1"
)

# Exclui avaliações já utilizadas na inferência anterior

avaliados = (
    spark.table("workspace.sabor_operacao.predicoes_reviews")
    .filter(F.col("lote_id") == "operacional_1")
    .select("Id").distinct()
)

lote = lote.join(avaliados, "Id", "left_anti")

# 2. Divide o lote cronologicamente

corte = lote.select(
    F.percentile_approx("time_unix", 0.7)
).first()[0]

def amostrar(df, n):
    amostra = (
        df.orderBy(F.xxhash64("Id", F.lit(42)), "Id")
        .limit(n)
        .select("Id", "texto_entrada", "alvo_baixa_nota")
    )
    assert amostra.count() == n
    return amostra.toPandas()

treino = pd.concat([
    amostrar(historico, 7000),
    amostrar(lote.filter(F.col("time_unix") <= corte), 3000)
], ignore_index=True)

validacao = amostrar(
    lote.filter(F.col("time_unix") > corte), 1500
)

X_train = treino["texto_entrada"]
y_train = treino["alvo_baixa_nota"]

X_val = validacao["texto_entrada"]
y_val = validacao["alvo_baixa_nota"]

# 3. Executa o único retreinamento

novo = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        dtype=np.float32
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

novo.fit(X_train, y_train)

# 4. Compara os modelos na mesma validação

limiar = float(atual.tags["limiar_classificacao"])

def avaliar(modelo):
    pred = (modelo.predict_proba(X_val)[:, 1] >= limiar).astype(int)

    return {
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0)
    }

m_atual = avaliar(modelo_atual)
m_novo = avaliar(novo)

promover = (
    m_novo["f1"] > m_atual["f1"]
    and m_novo["recall"] >= 0.70
)

# 5. Registra o retreinamento no MLflow

exemplo = X_train.iloc[:3].to_numpy(dtype=str)

with mlflow.start_run(run_name="retreinamento_operacional_1") as run:

    mlflow.log_params({
        "semente": 42,
        "n_treino": len(treino),
        "n_validacao": len(validacao),
        "corte_operacional_1": int(corte),
        "versao_referencia": atual.version,
        "limiar": limiar,
        "max_features": 10000,
        "promovido": promover
    })

    mlflow.log_metrics({
        f"atual_{k}": v for k, v in m_atual.items()
    })

    mlflow.log_metrics({
        f"novo_{k}": v for k, v in m_novo.items()
    })

    mlflow.log_dict({
        "treino": treino["Id"].astype(int).tolist(),
        "validacao": validacao["Id"].astype(int).tolist()
    }, "ids_amostras.json")

    info = mlflow.sklearn.log_model(
        sk_model=novo,
        name="modelo_retreinado",
        signature=infer_signature(
            exemplo, novo.predict(exemplo)
        ),
        input_example=exemplo,
        registered_model_name=MODELO
    )

    run_id = run.info.run_id

# 6. Atualiza aliases conforme a regra de promoção

versao_nova = info.registered_model_version

client.set_model_version_tag(
    MODELO, versao_nova,
    "limiar_classificacao", str(limiar)
)

client.set_registered_model_alias(
    MODELO, "candidate", versao_nova
)

if promover:
    client.set_registered_model_alias(
        MODELO, "champion", versao_nova
    )

# 7. Evidência consolidada

print("=== RETREINAMENTO ===")
print(f"Treino: {len(treino)} registros")
print(f"Validação: {len(validacao)} registros")
print(f"Versão anterior: {atual.version}")
print(f"Nova versão: {versao_nova}")
print(f"Run ID: {run_id}")

display(pd.DataFrame([
    {"versao": atual.version, **m_atual},
    {"versao": versao_nova, **m_novo}
]))

print("Decisão:", "PROMOVIDO" if promover else "NÃO PROMOVIDO")

champion = client.get_model_version_by_alias(
    MODELO, "champion"
)

print(f"Champion final: versão {champion.version}")
print("RETREINAMENTO CONCLUÍDO.")
