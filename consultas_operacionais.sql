
-- ===========================================================
-- SABOR EM FOCO
-- CONSULTAS OPERACIONAIS E DE AUDITORIA
-- QUESTÃO 2.1.1
-- ===========================================================

-- 1. SCHEMAS DO PROJETO

SHOW SCHEMAS IN workspace;

-- 2. OBJETOS DE DESENVOLVIMENTO

SHOW TABLES IN workspace.sabor_dev;

-- 3. OBJETOS OPERACIONAIS

SHOW TABLES IN workspace.sabor_operacao;

-- 4. TOTAL DE REGISTROS POR CAMADA

SELECT
    'Bronze' AS camada,
    COUNT(*) AS quantidade
FROM workspace.sabor_dev.bronze_reviews

UNION ALL

SELECT
    'Silver' AS camada,
    COUNT(*) AS quantidade
FROM workspace.sabor_dev.silver_reviews

UNION ALL

SELECT
    'Gold ML' AS camada,
    COUNT(*) AS quantidade
FROM workspace.sabor_dev.gold_reviews_ml;

-- 5. DISTRIBUIÇÃO DOS RECORTES TEMPORAIS

SELECT
    recorte_temporal,
    COUNT(*) AS quantidade,
    MIN(data_avaliacao) AS data_inicial,
    MAX(data_avaliacao) AS data_final,
    SUM(alvo_baixa_nota) AS notas_baixas
FROM workspace.sabor_dev.gold_reviews_ml
GROUP BY recorte_temporal
ORDER BY data_inicial;

-- 6. INDICADORES MENSAIS

SELECT
    mes_referencia,
    quantidade_avaliacoes,
    nota_media,
    proporcao_notas_1_2
FROM workspace.sabor_dev.gold_avaliacoes_mensais
ORDER BY mes_referencia DESC
LIMIT 12;

-- 7. RANKING DE PRODUTOS COM BAIXAS NOTAS
-- Mínimo de 20 avaliações

SELECT
    ProductId,
    quantidade_avaliacoes,
    nota_media,
    ROUND(proporcao_notas_1_2 * 100, 2)
        AS taxa_baixa_nota_pct
FROM workspace.sabor_dev.gold_avaliacoes_produto
WHERE quantidade_avaliacoes >= 20
ORDER BY
    proporcao_notas_1_2 DESC,
    quantidade_avaliacoes DESC,
    ProductId ASC
LIMIT 20;

-- 8. REJEIÇÕES SINTÉTICAS

SELECT
    motivo_rejeicao,
    COUNT(*) AS quantidade
FROM workspace.sabor_dev.quarentena_reviews_teste
GROUP BY motivo_rejeicao
ORDER BY quantidade DESC;

-- 9. CONTROLE DE VERSIONAMENTO

SELECT
    snapshot_id,
    tabela,
    versao_delta,
    timestamp_versao
FROM workspace.sabor_operacao.versoes_dados
ORDER BY
    snapshot_id,
    tabela;

-- ===========================================================
-- FIM DAS CONSULTAS
-- ===========================================================
