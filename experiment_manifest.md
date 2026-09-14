# Experiment Manifest

## Main Smoke Results

| Batch | Purpose | Command | Output |
|---|---|---|---|
| Batch 15 | Dense adaptive certificate regret on 48 year-round instances | `python -m src.trc_launch.run_smoke_batch15_dense_adaptive` | `results/trc_smoke/batch15_dense_adaptive/` |
| Batch 16 | Frontier baselines on 12 representative instances | `python -m src.trc_launch.run_smoke_batch16_frontier_baselines` | `results/trc_smoke/batch16_frontier_baselines/` |
| Batch 17 core20 | Twenty-instance open benchmark with regret, SAA, CVaR, mean-CVaR95, minimax, and active-scenario comparators | `python -m src.trc_launch.run_fullscale_batch17_open_benchmark --max-instances 20 --methods regret_portfolio_full adaptive_certificate_regret_dense saa_extensive cvar_extensive mean_cvar95_extensive minimax_extensive active_scenario_saa --time-limit 25 --oracle-time-limit 6 --data-dir data/trc_open_smoke/batch17_open_benchmark --output-dir results/trc_smoke/batch17_core20` | `results/trc_smoke/batch17_core20/` |
| Batch 17 frontier20 | Twenty-instance comparator benchmark with fixed-width certificates, supervised solution-space reduction, and iterative candidate generation | `python -m src.trc_launch.run_fullscale_batch17_open_benchmark --max-instances 20 --methods regret_portfolio_full adaptive_certificate_regret_dense fixed_k12 fixed_k16 fixed_k20 fixed_k36 iterative_candidate_generation ml_solution_reduction --time-limit 25 --oracle-time-limit 6 --data-dir data/trc_open_smoke/batch17_open_benchmark --output-dir results/trc_smoke/batch17_frontier20` | `results/trc_smoke/batch17_frontier20/` |
| Batch 17 combined20 | Merged twenty-instance comparator table used in the manuscript | generated from Batch 17 core20 and frontier20 metrics | `results/trc_smoke/batch17_combined20/` |
| Batch 18 weather audit | IEM ASOS/AWOS/METAR feature audit for selected open benchmark instances, 04:00-12:00 local weather window | `python -m src.trc_launch.run_weather_feature_audit --max-instances 20 --local-start-hour 4 --local-end-hour 12 --output results/trc_smoke/batch18_weather_audit/window_0400_1200_feature_audit.csv` | `results/trc_smoke/batch18_weather_audit/` |
| Batch 18 weather audit 48 | IEM ASOS/AWOS/METAR feature audit for 48 year-round benchmark instances, 04:00-12:00 local weather window | `python -m src.trc_launch.run_weather_feature_audit --selected-instances data/trc_open_smoke/batch14_year_round/selected_instances.csv --max-instances 48 --local-start-hour 4 --local-end-hour 12 --output results/trc_smoke/batch18_weather_audit/window_0400_1200_batch14_48_feature_audit.csv` | `results/trc_smoke/batch18_weather_audit/` |
| Batch 18 adverse weather smoke | One adverse-weather Bay Area instance with IEM weather-adjusted scenarios | `python -m src.trc_launch.run_fullscale_batch17_open_benchmark --max-instances 1 --methods regret_portfolio_full adaptive_certificate_regret_dense --scenario-count 12 --time-limit 20 --oracle-time-limit 5 --use-weather --data-dir data/trc_open_smoke/batch18_weather_adverse1 --output-dir results/trc_smoke/batch18_weather_adverse1` | `results/trc_smoke/batch18_weather_adverse1/` |

## Current Main Algorithm

`adaptive_certificate_regret_dense`

The method solves a regret portfolio on an adaptive certificate-filtered candidate graph. Dense instances use broader aircraft-side coverage.

## Main Baselines

- `regret_portfolio_full`
- `saa_extensive`
- `cvar_extensive`
- `mean_cvar95_extensive`
- `minimax_extensive`
- `active_scenario_saa`
- `fixed_k12`, `fixed_k16`, `fixed_k20`, `fixed_k36`
- `ml_solution_reduction`
- `iterative_candidate_generation`

## Full-Scale Target

- At least 300 carrier-region-date instances.
- 2025 full-year coverage.
- Seven to nine United States multi-airport regions.
- BTS schedule and tail continuity as the core data layer.
- IEM ASOS/AWOS/METAR weather features for the full scenario layer.
- Weather cache window: 04:00-12:00 airport-local time.
- Batch weather downloads must reuse `data/weather_asos/` cache and respect public endpoint rate limiting.

## Environmental conversion

| Batch | Purpose | Command | Output |
|---|---|---|---|
| Batch 35 environment | Convert 360-instance launch metrics to taxi fuel and CO2 with ICAO/EEA LTO factors | `python -m src.trc_launch.environmental_eval --mode convert` | `results/trc_smoke/batch35_environment/` and `article/trc_elsarticle/generated/environment_method_summary.csv` |
