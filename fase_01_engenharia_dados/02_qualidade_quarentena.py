# Databricks notebook source
# DBTITLE 1,1.2 validação de indentificadores

from pyspark.sql import functions as F

# 1.2.1.a - Validação dos identificadores obrigatórios

tabela_bronze = "workspace.sabor_dev.bronze_reviews"

df_qualidade = spark.table(tabela_bronze)

#fazer verificação
resultado_identificadores = (
    df_qualidade
    .agg(
        F.count("*").alias("total_registros"),

        F.count_if(
            F.col("Id").isNull() |
            (F.trim(F.col("Id")) == "")
        ).alias("id_invalido"),

        F.count_if(
            F.col("ProductId").isNull() |
            (F.trim(F.col("ProductId")) == "")
        ).alias("produto_invalido"),

        F.count_if(
            F.col("Id").isNull() |
            (F.trim(F.col("Id")) == "") |
            F.col("ProductId").isNull() |
            (F.trim(F.col("ProductId")) == "")
        ).alias("registros_com_identificador_invalido")
    )
)

print("=== 1.2.1.a - VALIDAÇÃO DE IDENTIFICADORES ===")

display(resultado_identificadores)


# COMMAND ----------

# DBTITLE 1,Validação Score

from pyspark.sql import functions as F

# 1.2.1.b - Validação do campo Score

df_score = spark.table(
    "workspace.sabor_dev.bronze_reviews"
)

# Verifica se o valor é inteiro e está entre 1 e 5
df_validacao_score = df_score.withColumn(
    "score_invalido",
    F.when(
        F.col("Score").isNull() |
        ~F.trim(F.col("Score")).rlike(r"^[1-5]$"),
        1
    ).otherwise(0)
)

# Consolida os resultados da validação
resultado_score = df_validacao_score.agg(
    F.count("*").alias("total_registros"),
    F.sum("score_invalido").alias("scores_invalidos"),
    F.min(
        F.when(
            F.col("score_invalido") == 0,
            F.col("Score").cast("int")
        )
    ).alias("menor_score_valido"),
    F.max(
        F.when(
            F.col("score_invalido") == 0,
            F.col("Score").cast("int")
        )
    ).alias("maior_score_valido")
)

print("=== 1.2.1.b - VALIDAÇÃO DO SCORE ===")
display(resultado_score)


# COMMAND ----------

# DBTITLE 1,Validar Time

from pyspark.sql import functions as F

# 1.2.1.c - Validação do campo Time

df_time = spark.table(
    "workspace.sabor_dev.bronze_reviews"
)

# Converte Time para número e posteriormente para data
df_validacao_time = (
    df_time
    .withColumn(
        "time_unix",
        F.expr("try_cast(trim(Time) AS BIGINT)")
    )
    .withColumn(
        "data_avaliacao",
        F.expr("try_cast(timestamp_seconds(time_unix) AS DATE)")
    )
)

# Consolida a validação
resultado_time = df_validacao_time.agg(
    F.count("*").alias("total_registros"),

    F.count_if(
        F.col("Time").isNull() |
        (F.trim("Time") == "") |
        F.col("time_unix").isNull() |
        F.col("data_avaliacao").isNull()
    ).alias("datas_invalidas"),

    F.min("data_avaliacao").alias("primeira_avaliacao"),
    F.max("data_avaliacao").alias("ultima_avaliacao")
)

print("=== 1.2.1.c - VALIDAÇÃO DO CAMPO TIME ===")
display(resultado_time)


# COMMAND ----------

# DBTITLE 1,Validação do campo Text

from pyspark.sql import functions as F

# 1.2.1.d - Validação de textos vazios após limpeza

df_texto = spark.table(
    "workspace.sabor_dev.bronze_reviews"
)

# Remove tags HTML e normaliza espaços
df_validacao_texto = (
    df_texto
    .withColumn(
        "texto_sem_html",
        F.regexp_replace(
            F.coalesce(F.col("Text"), F.lit("")),
            r"<[^>]*>",
            " "
        )
    )
    .withColumn(
        "texto_limpo",
        F.trim(
            F.regexp_replace(
                F.col("texto_sem_html"),
                r"\s+",
                " "
            )
        )
    )
)

# Identifica textos sem conteúdo após a limpeza
resultado_texto = df_validacao_texto.agg(
    F.count("*").alias("total_registros"),

    F.count_if(
        F.col("Text").isNull()
    ).alias("textos_nulos"),

    F.count_if(
        F.col("texto_limpo") == ""
    ).alias("textos_vazios_apos_limpeza"),

    F.count_if(
        (F.col("Text").isNotNull()) &
        (F.col("texto_limpo") == "")
    ).alias("textos_invalidos_nao_nulos")
)

print("=== 1.2.1.d - VALIDAÇÃO DOS TEXTOS ===")
display(resultado_texto)


# COMMAND ----------

# DBTITLE 1,Verificação de duplicidade por Id

from pyspark.sql import functions as F

# 1.2.1.f - Identificação de duplicatas por Id

df_duplicatas = spark.table(
    "workspace.sabor_dev.bronze_reviews"
)

# Agrupa avaliações pela chave Id
df_ids_repetidos = (
    df_duplicatas
    .groupBy("Id")
    .agg(F.count("*").alias("quantidade"))
    .filter(F.col("quantidade") > 1)
)

# Calcula os indicadores de duplicidade
resultado_duplicatas = df_ids_repetidos.agg(
    F.count("*").alias("ids_duplicados"),
    F.coalesce(
        F.sum(F.col("quantidade") - 1),
        F.lit(0)
    ).alias("registros_duplicados_excedentes")
)

print("=== 1.2.1.f - VALIDAÇÃO DE DUPLICATAS ===")
display(resultado_duplicatas)


# COMMAND ----------

# DBTITLE 1,Regras quarentena

from pyspark.sql import functions as F

# 1.2.1.e - Consolidação das regras de qualidade
df_origem = spark.table("workspace.sabor_dev.bronze_reviews")

# Expressões de validação
id_invalido = (
    F.col("Id").isNull() |
    (F.trim(F.col("Id")) == "")
)

produto_invalido = (
    F.col("ProductId").isNull() |
    (F.trim(F.col("ProductId")) == "")
)

score_invalido = (
    F.col("Score").isNull() |
    ~F.trim(F.col("Score")).rlike(r"^[1-5]$")
)

time_invalido = (
    F.col("Time").isNull() |
    F.expr("try_cast(trim(Time) AS BIGINT)").isNull()
)

texto_limpo = F.trim(
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

texto_invalido = (texto_limpo == "")

# Registra todas as violações encontradas em cada linha
df_regras_qualidade = df_origem.withColumn(
    "motivos_erro",
    F.array_compact(
        F.array(
            F.when(id_invalido, F.lit("ID_INVALIDO")),
            F.when(produto_invalido, F.lit("PRODUTO_INVALIDO")),
            F.when(score_invalido, F.lit("SCORE_INVALIDO")),
            F.when(time_invalido, F.lit("TIME_INVALIDO")),
            F.when(texto_invalido, F.lit("TEXTO_VAZIO"))
        )
    )
).withColumn(
    "status_qualidade",
    F.when(
        F.size("motivos_erro") > 0,
        "INVALIDO"
    ).otherwise("VALIDO")
)

# Resultado consolidado
resumo_qualidade = (
    df_regras_qualidade
    .groupBy("status_qualidade")
    .agg(F.count("*").alias("quantidade"))
)

print("=== 1.2.1.e - CONSOLIDAÇÃO DAS REGRAS ===")
display(resumo_qualidade)


# COMMAND ----------

# DBTITLE 1,Tabela delta quarentena

# 1.2.1.e - Persistência da quarentena em Delta

tabela_quarentena = "workspace.sabor_dev.quarentena_reviews"

# Seleciona somente registros com erros de conteúdo
df_quarentena = df_regras_qualidade.filter(
    F.col("status_qualidade") == "INVALIDO"
)

if not spark.catalog.tableExists(tabela_quarentena):
    (
        df_quarentena.write
        .format("delta")
        .mode("ignore")
        .saveAsTable(tabela_quarentena)
    )
    print(f"Tabela criada: {tabela_quarentena}")
else:
    print(f"Tabela já existente: {tabela_quarentena}")

# Consulta a estrutura e o total persistido
df_quarentena_delta = spark.table(tabela_quarentena)

print("=== ESTRUTURA DA QUARENTENA ===")
df_quarentena_delta.printSchema()

print("=== CONTAGEM DA QUARENTENA ===")
print(f"Registros na quarentena: {df_quarentena_delta.count()}")

display(
    df_quarentena_delta.select(
        "Id",
        "Score",
        "motivos_erro",
        "status_qualidade",
        "lote_id"
    ).limit(10)
)


# COMMAND ----------

# DBTITLE 1,1.2.2 - Criação do lote sintético

from pyspark.sql import functions as F

# 1.2.2.a - Criação de registros sintéticos para testes

dados_sinteticos = [
    (
        "SINT001", "PROD_TESTE", "9", "1303862400",
        "Nota inválida", "Produto com nota fora do intervalo."
    ),
    (
        "SINT002", "PROD_TESTE", "2", "1303862400",
        "Texto vazio", "<p> </p>"
    ),
    (
        "SINT003", "PROD_TESTE", "1", "1303862400",
        "Avaliação duplicada", "Produto chegou danificado."
    ),
    (
        "SINT003", "PROD_TESTE", "1", "1303862400",
        "Avaliação duplicada", "Produto chegou danificado."
    )
]

colunas_sinteticas = [
    "Id", "ProductId", "Score",
    "Time", "Summary", "Text"
]

# DataFrame exclusivo para testes de qualidade
df_sintetico = spark.createDataFrame(
    dados_sinteticos,
    schema=colunas_sinteticas
)

# Identificação explícita da origem dos registros
df_sintetico = (
    df_sintetico
    .withColumn("origem_registro", F.lit("SINTETICO"))
    .withColumn("lote_id", F.lit("teste_qualidade_001"))
)

print("=== 1.2.2.a - LOTE SINTÉTICO ===")
print(f"Total de registros de teste: {df_sintetico.count()}")

display(df_sintetico)


# COMMAND ----------

# DBTITLE 1,testar regras de qualidade

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# 1.2.2.a - Aplicação das regras ao lote sintético

# Validação do Score
score_invalido = (
    F.col("Score").isNull() |
    ~F.trim(F.col("Score")).rlike(r"^[1-5]$")
)

# Limpeza inicial do texto
texto_limpo = F.trim(
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

# Registra erros de conteúdo
df_teste_qualidade = (
    df_sintetico
    .withColumn(
        "motivos_erro",
        F.array_compact(
            F.array(
                F.when(
                    score_invalido,
                    F.lit("SCORE_INVALIDO")
                ),
                F.when(
                    texto_limpo == "",
                    F.lit("TEXTO_VAZIO")
                )
            )
        )
    )
    .withColumn(
        "status_conteudo",
        F.when(
            F.size("motivos_erro") > 0,
            "INVALIDO"
        ).otherwise("VALIDO")
    )
)

# Identificação independente de duplicatas
janela_duplicidade = Window.partitionBy("Id").orderBy(
    F.col("Summary"),
    F.col("Text")
)

df_teste_qualidade = (
    df_teste_qualidade
    .withColumn(
        "ocorrencia_id",
        F.row_number().over(janela_duplicidade)
    )
    .withColumn(
        "duplicata_excedente",
        F.col("ocorrencia_id") > 1
    )
)

print("=== 1.2.2.a - RESULTADO DAS REGRAS SINTÉTICAS ===")

display(
    df_teste_qualidade.select(
        "Id",
        "Score",
        "motivos_erro",
        "status_conteudo",
        "duplicata_excedente",
        "origem_registro"
    ).orderBy("Id", "duplicata_excedente")
)


# COMMAND ----------

# DBTITLE 1,Isolamento dos registros sintéticos

from pyspark.sql import functions as F

# 1.2.2.b - Persistência isolada dos testes sintéticos

tabela_teste = "workspace.sabor_dev.quarentena_reviews_teste"

# Seleciona erros de conteúdo ou duplicatas excedentes
df_rejeitados_teste = (
    df_teste_qualidade
    .filter(
        (F.col("status_conteudo") == "INVALIDO") |
        (F.col("duplicata_excedente") == True)
    )
    .withColumn(
        "motivo_rejeicao",
        F.when(
            F.col("duplicata_excedente"),
            F.lit("DUPLICATA")
        ).otherwise(F.lit("ERRO_CONTEUDO"))
    )
)

# Cria a tabela de teste sem sobrescrever dados existentes
if not spark.catalog.tableExists(tabela_teste):
    (
        df_rejeitados_teste.write
        .format("delta")
        .mode("ignore")
        .saveAsTable(tabela_teste)
    )
    print(f"Tabela criada: {tabela_teste}")
else:
    print(f"Tabela já existente: {tabela_teste}")

# Confere o isolamento entre as tabelas
df_teste_delta = spark.table(tabela_teste)

resultado_isolamento = spark.createDataFrame(
    [
        (
            "Bronze original",
            spark.table("workspace.sabor_dev.bronze_reviews").count()
        ),
        (
            "Quarentena original",
            spark.table("workspace.sabor_dev.quarentena_reviews").count()
        ),
        (
            "Quarentena sintética",
            df_teste_delta.count()
        )
    ],
    ["tabela", "total_registros"]
)

print("=== 1.2.2.b - VALIDAÇÃO DO ISOLAMENTO ===")
display(resultado_isolamento)

print("=== REGISTROS SINTÉTICOS REJEITADOS ===")
display(
    df_teste_delta.select(
        "Id",
        "motivos_erro",
        "duplicata_excedente",
        "motivo_rejeicao",
        "origem_registro",
        "lote_id"
    )
)


# COMMAND ----------


# 1.2.2.b - Conferência final do isolamento

tabelas = {
    "Bronze original": "workspace.sabor_dev.bronze_reviews",
    "Quarentena original": "workspace.sabor_dev.quarentena_reviews",
    "Quarentena sintética": "workspace.sabor_dev.quarentena_reviews_teste"
}

for descricao, tabela in tabelas.items():
    quantidade = spark.table(tabela).count()
    print(f"{descricao}: {quantidade:,}")
