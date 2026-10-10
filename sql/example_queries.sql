-- Run against outputs/monitoring.db, e.g.  sqlite3 outputs/monitoring.db < sql/example_queries.sql

-- 1. Quarters that breached any threshold
SELECT period, psi, gini, actual_to_predicted, overall_rag
FROM performance
WHERE overall_rag <> 'Green'
ORDER BY period;

-- 2. The feature with the highest CSI in each quarter
SELECT period, feature, MAX(csi) AS csi
FROM csi
GROUP BY period
ORDER BY period;

-- 3. Deciles where the model under-predicts defaults by more than 20%
SELECT period, decile, mean_predicted_pd, actual_default_rate
FROM calibration
WHERE actual_default_rate > 1.2 * mean_predicted_pd
ORDER BY period, decile;
