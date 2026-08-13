"""Neural forecasting base package."""

from iqrp.app.forecasting.neural.base.neural_model import NeuralForecastModel
from iqrp.app.forecasting.neural.base.trainer import NeuralTrainer

__all__ = ["NeuralForecastModel", "NeuralTrainer"]
