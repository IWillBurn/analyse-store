import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing

trends = [False, "add", "mul"]
seasonals = [False, "add", "mul"]
use_boxcox = [False, True]
init_methods = ['estimated', 'concentrated', 'heuristic']


class ExponentialSmoothingModel:

    @staticmethod
    def full_space() -> dict:
        """
        Возвращает поддерживаемые значения параметров.
        """
        space_data = {
            'trends': trends,
            'seasonals': seasonals,
            'init_methods': init_methods,
            'use_boxcox': use_boxcox,
        }
        return space_data

    def __init__(self, space: list):
        self.space = space
        self.models = {}

    def fit(self, train: pd.Series) -> None:
        """
        Обучает модели с параметрами из space.
        """
        self.models = {}
        for params in self.space:
            model = ExponentialSmoothing(
                train,
                trend=params['trend'],
                seasonal=params['seasonal'],
                seasonal_periods=params['seasonal_periods'],
                initialization_method=params['init'],
                use_boxcox=params['use_boxcox'],
            ).fit()
            self.models[params['name']] = model

    def summary(self) -> None:
        """
        Выводит информацию по всем обученным моделям.
        """
        for name, model in self.models.items():
            print(name)
            print(model.summary())

    def predict(self, count: int, index: pd.Index) -> dict:
        """
        Возвращает предсказания всех моделей на отрезок [0, train + count).
        """
        predicts = {}
        for name, model in self.models.items():
            forecast = model.forecast(count)
            predicts[name] = pd.Series(forecast.to_list(), index=index)
        return predicts
