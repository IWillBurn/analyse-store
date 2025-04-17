import pandas as pd
from statsmodels.tsa.api import UnobservedComponents
from typing import Union

levels = [
    'irregular',
    'fixed intercept',
    'deterministic constant',
    'local level',
    'random walk',
    'fixed slope',
    'deterministic trend',
    'local linear deterministic trend',
    'random walk with drift',
    'local linear trend',
    'smooth trend',
    'random trend',
]
trends = [False, True, 'fixed']
seasonals = [None, 12, 'dummy', 'trigonometric']
cycles = [False, True, 1]
irregulars = [False, True]
stochastic_levels = [False, True]
stochastic_trends = [False, True]
damped_cycles = [False, True]


class UnobservedComponentsModel:

    @staticmethod
    def get_space() -> dict:
        """
        Возвращает поддерживаемые значения параметров.
        """
        space = {
            "levels": levels,
            "trends": trends,
            "seasonals": seasonals,
            "cycles": cycles,
            "irregulars": irregulars,
            "stochastic_levels": stochastic_levels,
            "stochastic_trends": stochastic_trends,
            "damped_cycles": damped_cycles,
        }
        return space

    def __init__(self, space: list):
        self.space = space
        self.models = {}
        self.train_len = 0
        self.exog = None

    def fit(self, train: pd.Series, exog: Union[pd.DataFrame, None]) -> None:
        """
        Обучает модели с параметрами из space и указанными exog.
        """
        self.models = {}
        self.train_len = len(train)
        self.exog = exog
        for params in self.space:
            model = UnobservedComponents(train['cnt'],
                                         level=params['level'],
                                         trend=params['trend'],
                                         seasonal=params['seasonal'],
                                         exog=exog,
                                         cycle=params['cycle'],
                                         irregular=params['irregular'],
                                         stochastic_level=params['stochastic_level'],
                                         stochastic_trend=params['stochastic_trend'],
                                         damped_cycle=params['damped_cycle'],
                                         initialization_method='heuristic')
            model = model.fit()
            self.models[params['name']] = model

    def summary(self) -> None:
        """
        Выводит информацию по всем обученным моделям.
        """
        for name, model in self.models.items():
            print(name)
            print(model.summary())

    def predict(self, count: int, exog: Union[pd.DataFrame, None], index: pd.Index) -> dict:
        """
        Возвращает предсказания всех моделей на отрезок [0, train + count).
        """
        sum_exog = None
        if self.exog != None and exog != None:
            sum_exog = pd.concat(self.exog, exog)
        predicts = {}
        for name, model in self.models.items():
            forecast = model.predict(start=self.train_len,
                                     end=count + self.train_len - 1,
                                     exog=sum_exog,
                                     dynamic=False)
            predicts[name] = pd.Series(forecast.to_list(), index=index)
        return predicts
