# LightGBM in IQRP

Model name: `lightgbm`  
Class: `LightGBMForecastModel`  
Trainer: `LightGBMTrainer`

Supports regression, classification, and quantile objectives (`objective=quantile`, `alpha=...`).

Device selection via `hyperparameters.device` (`cpu` / `gpu`). Verbosity is silenced for production logs.
