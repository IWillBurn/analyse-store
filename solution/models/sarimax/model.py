from typing import Union

import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SarimaxModel:
    def __init__(self,
                 order: tuple = (0, 0, 0),
                 seasonal_order: tuple = (0, 0, 0),
                 auto_p: list = None,
                 auto_q: list = None,
                 auto_P: list = None,
                 auto_Q: list = None,
                 auto_d: int = 0,
                 auto_D: int = 0,
                 auto_seasonal_period: int = 0,
    ):
        self.name = "SARIMAX"
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_p = auto_p
        self.auto_q = auto_q
        self.auto_P = auto_P
        self.auto_Q = auto_Q
        self.auto_d = auto_d
        self.auto_D = auto_D
        self.auto_seasonal_period = auto_seasonal_period
        self.model = None
        self.exog = None
        self.train = None

    def fit(self, train: pd.Series, exog: Union[pd.DataFrame, None]) -> None:
        """
        Обучает модель по параметрам с указанными exog.
        """
        self.exog = exog
        self.train = train
        model = SARIMAX(train, exog=exog, order=self.order, seasonal_order=self.seasonal_order)
        model = model.fit()
        self.model = model

    def fit_auto(self, train: pd.Series, exog: Union[pd.DataFrame, None]) -> None:
        """
        Атоматически подбирает параметры и обучает SARIMA в диапозоне.
        """
        self.exog = None
        self.train = train
        self.model = pm.auto_arima(
            train,
            exogenous=exog,
            start_p=self.auto_p[0],
            start_q=self.auto_q[0],
            max_p=self.auto_p[1],
            max_q=self.auto_q[1],
            m=self.auto_seasonal_period,
            seasonal=self.auto_seasonal_period != 1,
            d=self.auto_d,
            D=self.auto_D,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

    def summary(self):
        """
        Выводит информацию о полученной модели.
        """
        print(self.model.summary())

    def predict(self, count: int, exog: Union[pd.DataFrame, None], index: pd.Index) -> pd.Series:
        """
        Возвращает предсказания модели на отрезок [0, train + count).
        """
        use_exog = None
        if exog != None and self.exog != None:
            use_exog = exog
        train_predict = self.model.predict(start=0, end=len(self.train) - 1, dynamic=False, exog=self.exog)
        forecast_predictions = self.model.predict(start=len(self.train), end=len(self.train) + count - 1, dynamic=True, exog=use_exog)
        predict = train_predict.to_list() + forecast_predictions.to_list()
        return pd.Series(predict, index=index)

    def predict_auto(self, count: int, exog: Union[pd.DataFrame, None], index: pd.Index) -> pd.Series:
        """
        Возвращает предсказание лучше модели, подобранной и обученной автоматически.
        """
        use_exog = None
        if exog != None and self.exog != None:
            use_exog = exog
        forecast = self.model.predict(n_periods=count, exogenous=use_exog)
        return pd.Series(forecast.to_list(), index=index)
