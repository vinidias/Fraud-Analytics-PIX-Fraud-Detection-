-- ============================================================
-- Semana 2 — Padrões clássicos de fraude em SQL
-- Contexto: identidade mobile, device, localização
-- ============================================================

-- 1. Um device usado por múltiplos clientes (account takeover / farm)
SELECT
    device_id,
    COUNT(DISTINCT customer_id) AS n_customers,
    COUNT(*)                    AS n_tx
FROM transactions
GROUP BY device_id
HAVING COUNT(DISTINCT customer_id) > 3
ORDER BY n_customers DESC
LIMIT 50;

-- 2. Cliente usando múltiplos devices em curto intervalo (device sharing)
WITH dev_seq AS (
    SELECT
        customer_id,
        device_id,
        event_time,
        LAG(device_id) OVER (PARTITION BY customer_id ORDER BY event_time) AS prev_device,
        LAG(event_time) OVER (PARTITION BY customer_id ORDER BY event_time) AS prev_time
    FROM transactions
)
SELECT *
FROM dev_seq
WHERE device_id <> prev_device
  AND event_time - prev_time < INTERVAL '10 minutes'
ORDER BY customer_id, event_time;

-- 3. Transações em locais geograficamente impossíveis (impossible travel)
WITH geo_seq AS (
    SELECT
        customer_id,
        event_time,
        lat,
        lon,
        LAG(event_time) OVER (PARTITION BY customer_id ORDER BY event_time) AS prev_time,
        LAG(lat)        OVER (PARTITION BY customer_id ORDER BY event_time) AS prev_lat,
        LAG(lon)        OVER (PARTITION BY customer_id ORDER BY event_time) AS prev_lon
    FROM transactions
)
SELECT
    customer_id,
    event_time,
    ROUND(distance_km_haversine(prev_lat, prev_lon, lat, lon)::numeric, 1) AS km,
    ROUND(EXTRACT(EPOCH FROM (event_time - prev_time))::numeric / 3600, 2) AS horas,
    ROUND(
        distance_km_haversine(prev_lat, prev_lon, lat, lon)
        / NULLIF(EXTRACT(EPOCH FROM (event_time - prev_time)) / 3600, 0)
    , 1) AS km_por_hora
FROM geo_seq
WHERE prev_lat IS NOT NULL
  AND distance_km_haversine(prev_lat, prev_lon, lat, lon) > 500
  AND EXTRACT(EPOCH FROM (event_time - prev_time)) / 3600 < 2;

-- 4. Cartão testado com múltiplos valores pequenos antes de compra grande
WITH small_tx AS (
    SELECT
        card_id,
        event_time,
        amount,
        COUNT(*) FILTER (WHERE amount <= 5) OVER (
            PARTITION BY card_id
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
        ) AS n_small_1h
    FROM transactions
)
SELECT *
FROM small_tx
WHERE amount > 500 AND n_small_1h >= 3
ORDER BY card_id, event_time;
