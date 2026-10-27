-- Run against outputs/monitoring.db after src/run_ecl.py

-- 1. Coverage by stage
SELECT stage, loans, ROUND(ead, 0) AS ead, ROUND(ecl, 0) AS ecl, ROUND(coverage, 4) AS coverage
FROM ecl_stages ORDER BY stage;

-- 2. Top 10 portfolio segments by probability-weighted ECL (aggregation view)
SELECT purpose, home_ownership, loans, ROUND(ecl, 0) AS ecl, ROUND(coverage, 4) AS coverage,
       ROUND(stage2_share, 3) AS stage2_share
FROM ecl_aggregation ORDER BY ecl DESC LIMIT 10;

-- 3. How much each scenario adds over baseline
SELECT s.scenario, ROUND(s.ecl, 0) AS ecl, ROUND(s.ecl / b.ecl - 1, 3) AS uplift_vs_baseline
FROM ecl_scenarios s CROSS JOIN (SELECT ecl FROM ecl_scenarios WHERE scenario = 'baseline') b;
