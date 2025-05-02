import pandas as pd
from catboost import CatBoostRegressor
from statsmodels.tsa.api import ExponentialSmoothing

class CatBoostModel:
    def __init__(self, fields: list):
        self.name = "CATBOOST"
        self.model = None
        self.fields = fields

    def fit(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """
        Обучает модель.
        """
        self.model = CatBoostRegressor(
            verbose=100,
            random_seed=23523,
            depth=8,
            loss_function="RMSE"
        )
        train['cnt'] = train['cnt'].fillna(train['cnt'].mean())
        self.model.fit(train.drop(['cnt'], axis=1)[self.fields].iloc[-365::], train['cnt'][-365::], eval_set=(test.drop(['cnt'], axis=1)[self.fields], test['cnt']), plot=True)

    def summary(self) -> None:
        """
        Выводит информацию по всем обученным моделям.
        """
        pass

    def predict(self, test: pd.DataFrame, index: pd.Index) -> pd.Series:
        """
        Возвращает предсказания всех моделей на отрезок [0, train + count).
        """
        predict = self.model.predict(test[self.fields])
        return pd.Series(predict, index=index)
