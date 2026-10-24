-- ============================================================
-- Semana 2 — Window functions para análise de fraude
-- Foco: velocity checks, comportamento por cliente, anomalias
-- ============================================================

-- 1. Ranking de transações por valor dentro de cada cliente
SELECT
    tx_id,
    customer_id,
    amount,
    event_time,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY amount DESC) AS rank_valor,
    RANK()       OVER (PARTITION BY customer_id ORDER BY amount DESC) AS rank_valor_gaps
FROM transactions;

-- 2. Tempo entre transações consecutivas do mesmo cliente (velocity)
SELECT
    tx_id,
    customer_id,
    event_time,
    LAG(event_time) OVER (PARTITION BY customer_id ORDER BY event_time) AS prev_tx_time,
    EXTRACT(EPOCH FROM (
        event_time - LAG(event_time) OVER (PARTITION BY customer_id ORDER BY event_time)
    )) AS seconds_since_prev,
    is_fraud
FROM transactions
ORDER BY customer_id, event_time;

-- 3. Detectar "burst" — 3+ transações em menos de 5 minutos
WITH tx_velocity AS (
    SELECT
        tx_id,
        customer_id,
        event_time,
        is_fraud,
        COUNT(*) OVER (
            PARTITION BY customer_id
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '5 minutes' PRECEDING AND CURRENT ROW
        ) AS tx_last_5min
    FROM transactions
)
SELECT *
FROM tx_velocity
WHERE tx_last_5min >= 3
ORDER BY customer_id, event_time;

-- 4. Valor acumulado e % do total por cliente
SELECT
    tx_id,
    customer_id,
    amount,
    SUM(amount) OVER (PARTITION BY customer_id ORDER BY event_time) AS cumulative_amount,
    SUM(amount) OVER (PARTITION BY customer_id)                     AS total_amount,
    ROUND(
        100.0 * SUM(amount) OVER (PARTITION BY customer_id ORDER BY event_time)
              / SUM(amount) OVER (PARTITION BY customer_id)
    , 2) AS pct_acumulado
FROM transactions;

-- 5. Desvio do valor em relação à média do cliente (z-score simples)
WITH stats AS (
    SELECT
        tx_id,
        customer_id,
        amount,
        is_fraud,
        AVG(amount) OVER (PARTITION BY customer_id) AS mean_customer,
        STDDEV(amount) OVER (PARTITION BY customer_id) AS std_customer
    FROM transactions
)
SELECT
    *,
    ROUND((amount - mean_customer) / NULLIF(std_customer, 0), 2) AS z_score
FROM stats
WHERE ABS((amount - mean_customer) / NULLIF(std_customer, 0)) > 3
ORDER BY z_score DESC;

-- 6. Chargeback rate por merchant (últimos 30 dias)
SELECT
    merchant_id,
    COUNT(*)                                                   AS n_tx,
    SUM(CASE WHEN is_chargeback THEN 1 ELSE 0 END)             AS n_chargeback,
    ROUND(100.0 * SUM(CASE WHEN is_chargeback THEN 1 ELSE 0 END)
          / COUNT(*), 4)                                        AS chargeback_rate_pct
FROM transactions
WHERE event_time >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY merchant_id
HAVING COUNT(*) >= 50
ORDER BY chargeback_rate_pct DESC
LIMIT 20;

-- 7. Comparação com baseline do merchant (identificar anomalias)
WITH merchant_baseline AS (
    SELECT
        merchant_id,
        AVG(amount)    AS merchant_avg,
        STDDEV(amount) AS merchant_std
    FROM transactions
    GROUP BY merchant_id
)
SELECT
    t.tx_id,
    t.merchant_id,
    t.amount,
    m.merchant_avg,
    ROUND((t.amount - m.merchant_avg) / NULLIF(m.merchant_std, 0), 2) AS z_vs_merchant
FROM transactions t
JOIN merchant_baseline m USING (merchant_id)
WHERE t.amount > m.merchant_avg + 3 * m.merchant_std
ORDER BY z_vs_merchant DESC
LIMIT 100;
