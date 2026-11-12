# Databricks notebook source

from datetime import date
from pyspark.sql import functions as F

# ============================================================
# SABOR EM FOCO
# FASE 2 - QUESTÃO 2.1
# CONFIGURAÇÃO CENTRALIZADA DO PROJETO
# ============================================================

# ============================================================
# 1. CONFIGURAÇÃO CENTRAL
# ============================================================

CONFIG = {

    # Identificação do projeto
    "projeto": "Sabor em Foco",
    "versao_config": "1.0.0",

    # Unity Catalog
    "catalogo": "workspace",
    "schema_dev": "sabor_dev",
    "schema_operacao": "sabor_operacao",

    # Origem dos dados
    "volume": "dados_brutos",
    "arquivo": "Reviews.csv",

    # Reprodutibilidade
    "semente": 42,

    # Cortes cronológicos validados na Fase 1
    "corte_desenvolvimento_unix": 1336521600,
    "corte_operacional_1_unix": 1344384000,

    "data_corte_desenvolvimento": "2012-05-09",
    "data_corte_operacional_1": "2012-08-08",

    # Proporções dos recortes históricos
    "proporcao_desenvolvimento": 0.80,
    "proporcao_operacional_1": 0.10,
    "proporcao_operacional_2": 0.10,

    # Divisão interna do desenvolvimento para ML
    "proporcao_treino": 0.70,
    "proporcao_validacao": 0.15,
    "proporcao_teste": 0.15,

    # Limites de amostras
    "amostra_engenharia_inicial": 10000,
    "amostra_treinamento": 10000,
    "limite_treinamento": 50000,

    "amostra_retreinamento": 10000,
    "limite_retreinamento": 50000,

    "amostra_inferencia": 1000,
    "limite_inferencia": 5000,

    # Limites de IA Generativa
    "amostra_sentimento": 50,
    "amostra_resumo": 10,
    "limite_palavras_resumo": 40,

    # Machine Learning
    "max_features_tfidf": 10000,
    "recall_minimo_classe_1": 0.70,

    # Monitoramento
    "limite_drift_texto_pct": 30,

    # Indicadores Gold
    "volume_minimo_produto": 20,

    # MLflow
    "nome_experimento": "/Shared/sabor_em_foco",

    # Model Registry
    "nome_modelo_baseline": "modelo_baseline",
    "nome_modelo_candidato": "modelo_baixa_nota",

    # Snapshot de referência da Fase 1
    "snapshot_dados": "fase1_questao1_4_v1"
}

# ============================================================
# 2. CAMINHOS DERIVADOS DA CONFIGURAÇÃO
# ============================================================

CATALOGO = CONFIG["catalogo"]
SCHEMA_DEV = CONFIG["schema_dev"]
SCHEMA_OPERACAO = CONFIG["schema_operacao"]

BASE_DEV = f"{CATALOGO}.{SCHEMA_DEV}"
BASE_OPERACAO = f"{CATALOGO}.{SCHEMA_OPERACAO}"

CAMINHO_VOLUME = (
    f"/Volumes/{CATALOGO}/{SCHEMA_DEV}/"
    f"{CONFIG['volume']}"
)

CAMINHO_CSV = (
    f"{CAMINHO_VOLUME}/{CONFIG['arquivo']}"
)

TABELAS = {

    # Desenvolvimento
    "bronze": f"{BASE_DEV}.bronze_reviews",
    "silver": f"{BASE_DEV}.silver_reviews",
    "quarentena": f"{BASE_DEV}.quarentena_reviews",
    "quarentena_teste": f"{BASE_DEV}.quarentena_reviews_teste",

    "gold_mensal": f"{BASE_DEV}.gold_avaliacoes_mensais",
    "gold_produto": f"{BASE_DEV}.gold_avaliacoes_produto",
    "gold_ml": f"{BASE_DEV}.gold_reviews_ml",

    # Operação
    "versoes_dados": f"{BASE_OPERACAO}.versoes_dados",
    "logs_execucao": f"{BASE_OPERACAO}.logs_execucao",
    "predicoes": f"{BASE_OPERACAO}.predicoes_reviews",
    "monitoramento": f"{BASE_OPERACAO}.monitoramento_reviews",
    "metricas_modelo": f"{BASE_OPERACAO}.metricas_modelo",
    "resultados_ia": f"{BASE_OPERACAO}.resultados_ia"
}

MODELOS = {
    "baseline": (
        f"{BASE_DEV}.{CONFIG['nome_modelo_baseline']}"
    ),
    "candidato": (
        f"{BASE_DEV}.{CONFIG['nome_modelo_candidato']}"
    )
}

RECORTES = {
    "desenvolvimento": 454533,
    "operacional_1": 57001,
    "operacional_2": 56920
}

# ============================================================
# 3. VALIDAÇÃO DA CONFIGURAÇÃO
# ============================================================

def validar_configuracao():

    assert CONFIG["schema_dev"] != CONFIG["schema_operacao"]

    assert CONFIG["corte_desenvolvimento_unix"] < (
        CONFIG["corte_operacional_1_unix"]
    )

    assert date.fromisoformat(
        CONFIG["data_corte_desenvolvimento"]
    ) < date.fromisoformat(
        CONFIG["data_corte_operacional_1"]
    )

    assert abs(
        CONFIG["proporcao_desenvolvimento"]
        + CONFIG["proporcao_operacional_1"]
        + CONFIG["proporcao_operacional_2"]
        - 1.0
    ) < 1e-9

    assert abs(
        CONFIG["proporcao_treino"]
        + CONFIG["proporcao_validacao"]
        + CONFIG["proporcao_teste"]
        - 1.0
    ) < 1e-9

    assert (
        1 <= CONFIG["amostra_treinamento"]
        <= CONFIG["limite_treinamento"]
        <= 50000
    )

    assert (
        1 <= CONFIG["amostra_retreinamento"]
        <= CONFIG["limite_retreinamento"]
        <= 50000
    )

    assert (
        1 <= CONFIG["amostra_inferencia"]
        <= CONFIG["limite_inferencia"]
        <= 5000
    )

    assert 1 <= CONFIG["amostra_sentimento"] <= 50
    assert 1 <= CONFIG["amostra_resumo"] <= 10

    assert CONFIG["volume_minimo_produto"] >= 20
    assert 0 <= CONFIG["recall_minimo_classe_1"] <= 1
    assert CONFIG["limite_drift_texto_pct"] > 0

    return True


validar_configuracao()

# ============================================================
# 4. FUNÇÕES COMPARTILHADAS
# ============================================================

def classificar_recorte(coluna_time):

    return (
        F.when(
            coluna_time.isNull(),
            F.lit("sem_recorte")
        )
        .when(
            coluna_time <= CONFIG["corte_desenvolvimento_unix"],
            F.lit("desenvolvimento")
        )
        .when(
            coluna_time <= CONFIG["corte_operacional_1_unix"],
            F.lit("operacional_1")
        )
        .otherwise(F.lit("operacional_2"))
    )


def limpar_texto(coluna):

    return F.trim(
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


def gerar_alvo(coluna_score):

    return (
        F.when(
            coluna_score.isin(1, 2),
            F.lit(1)
        )
        .when(
            coluna_score.isin(3, 4, 5),
            F.lit(0)
        )
        .otherwise(F.lit(None).cast("int"))
    )

# ============================================================
# 5. VERIFICAÇÃO DOS SCHEMAS E DAS TABELAS
# ============================================================

print("=== 2.1.1.a - CONFIGURAÇÃO CENTRALIZADA ===")

for chave, valor in CONFIG.items():
    print(f"{chave}: {valor}")

print(f"\nVolume: {CAMINHO_VOLUME}")
print(f"Arquivo CSV: {CAMINHO_CSV}")

print("\n=== 2.1.1.b - SEPARAÇÃO DOS SCHEMAS ===")

schemas = spark.sql(
    f"SHOW SCHEMAS IN {CATALOGO}"
).collect()

nomes_schemas = {linha[0] for linha in schemas}

assert SCHEMA_DEV in nomes_schemas, (
    "Schema de desenvolvimento não encontrado."
)

assert SCHEMA_OPERACAO in nomes_schemas, (
    "Schema operacional não encontrado."
)

print(f"Desenvolvimento: {BASE_DEV} - OK")
print(f"Operação: {BASE_OPERACAO} - OK")

# Verifica o arquivo existente no Volume
arquivos_volume = dbutils.fs.ls(CAMINHO_VOLUME)

assert any(
    arquivo.name == CONFIG["arquivo"]
    for arquivo in arquivos_volume
), "Arquivo Reviews.csv não encontrado no Volume."

print(f"Arquivo {CONFIG['arquivo']}: ENCONTRADO")

# Identifica as tabelas existentes e as futuras
registros_tabelas = []

for nome, tabela in TABELAS.items():

    existe = spark.catalog.tableExists(tabela)

    registros_tabelas.append((
        nome,
        tabela,
        "EXISTENTE" if existe else "PLANEJADA"
    ))

df_controle_tabelas = spark.createDataFrame(
    registros_tabelas,
    [
        "responsabilidade",
        "tabela",
        "situacao"
    ]
)

display(df_controle_tabelas)

# Tabelas obrigatórias já criadas na Fase 1
tabelas_obrigatorias = [
    "bronze",
    "silver",
    "quarentena",
    "quarentena_teste",
    "gold_mensal",
    "gold_produto",
    "gold_ml",
    "versoes_dados"
]

for nome in tabelas_obrigatorias:

    assert spark.catalog.tableExists(
        TABELAS[nome]
    ), f"Tabela obrigatória ausente: {TABELAS[nome]}"

# ============================================================
# 6. RESUMO FINAL
# ============================================================

print("\n=== RESUMO FINAL - QUESTÃO 2.1 ===")
print(f"Projeto: {CONFIG['projeto']}")
print(f"Versão da configuração: {CONFIG['versao_config']}")
print("Parâmetros centralizados: OK")
print("Cortes temporais configurados: OK")
print("Limites de amostras: OK")
print("Semente e experimento MLflow: OK")
print("Separação desenvolvimento/operação: OK")
print("Volume e arquivo de origem: OK")
print("Tabelas existentes da Fase 1: OK")
print("Funções compartilhadas: OK")
print("\nVALIDAÇÃO APROVADA")
print("Nenhuma tabela foi alterada.")
