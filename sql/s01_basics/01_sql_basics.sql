-- ============================================================
-- Semana 1-2 — SQL básico para análise de fraude
-- Dataset de prática: jaffle_shop (dbt) ou paySim
-- Teste com DuckDB: duckdb < sql/s01_basics/01_sql_basics.sql
-- ============================================================

-- 1. Contagem e taxa de fraude por classe
SELECT
    is_fraud,
    COUNT(*)                          AS n_tx,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 4) AS pct
FROM transactions
GROUP BY is_fraud
ORDER BY is_fraud;

-- 2. Ticket médio por classe
SELECT
    is_fraud,
    COUNT(*)            AS n_tx,
    ROUND(AVG(amount), 2) AS avg_amount,
    ROUND(MIN(amount), 2) AS min_amount,
    ROUND(MAX(amount), 2) AS max_amount
FROM transactions
GROUP BY is_fraud;

-- 3. Top 10 clientes por volume transacionado
SELECT
    customer_id,
    COUNT(*)              AS n_tx,
    ROUND(SUM(amount), 2) AS total_amount
FROM transactions
GROUP BY customer_id
ORDER BY total_amount DESC
LIMIT 10;

-- 4. Taxa de fraude por faixa de valor
WITH faixas AS (
    SELECT
        CASE
            WHEN amount < 10   THEN '00-10'
            WHEN amount < 50   THEN '10-50'
            WHEN amount < 200  THEN '50-200'
            WHEN amount < 1000 THEN '200-1k'
            ELSE '1k+'
        END AS faixa,
        is_fraud
    FROM transactions
)
SELECT
    faixa,
    COUNT(*)                                    AS n_tx,
    SUM(is_fraud)                               AS n_fraude,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)  AS taxa_fraude_pct
FROM faixas
GROUP BY faixa
ORDER BY faixa;

-- 5. Fraude por hora do dia
SELECT
    EXTRACT(hour FROM event_time)              AS hora,
    COUNT(*)                                   AS n_tx,
    SUM(is_fraud)                              AS n_fraude,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS taxa_fraude_pct
FROM transactions
GROUP BY hora
ORDER BY taxa_fraude_pct DESC;
