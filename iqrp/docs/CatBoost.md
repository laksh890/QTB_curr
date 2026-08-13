# CatBoost in IQRP

Model name: `catboost`  
Class: `CatBoostForecastModel`  
Trainer: `CatBoostTrainer`

Uses ordered boosting with `allow_writing_files=False` for clean CI/server environments. Quantile loss: `Quantile:alpha=...`. GPU via `task_type=GPU` when `device` is `gpu`/`cuda`.
