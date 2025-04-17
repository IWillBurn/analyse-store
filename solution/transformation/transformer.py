import numpy as np
import pandas as pd
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sklearn.preprocessing import PowerTransformer


class Transformer:
    def __init__(self, solution):
        self.solution = solution

    def complete_zeroes(self, data: pd.DataFrame) -> (int, int):
        """
        Заполняет нули в данных через первый ненулевой элемент до или первый ненулевой элемент после.
        """
        was_zeros = (data['cnt'] == 0).sum()
        data['cnt'] = data['cnt'].replace(0, np.nan).ffill()
        now_zeros = (data['cnt'] == 0).sum()
        return was_zeros, now_zeros

    def yeojohnson(self, data: pd.Series) -> (PowerTransformer, pd.Series):
        """
        Применяет преобразование Йео-Джонсона.
        """
        transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        return transformer, pd.Series(transformer.fit_transform(data.to_frame()).flatten(), index=data.index)

    def yeojohnson_reverse(self, transformer: PowerTransformer, data: pd.Series) -> pd.Series:
        """
        Отменяет преобразование Йео-Джонсона.
        """
        return pd.Series(transformer.inverse_transform(data.to_frame()).flatten(), index=data.index)

    def boxcox(self, data: pd.Series) -> (BoxCoxTransformer, pd.Series):
        """
        Применяет преобразование Бокса-Кокса.
        """
        transformer = PowerTransformer(method="box-cox")
        return transformer, pd.Series(transformer.fit_transform(data.to_frame()).flatten(), index=data.index)

    def boxcox_reverse(self, transformer: BoxCoxTransformer, data: pd.Series) -> pd.Series:
        """
        Отменяет преобразование Бокса-Кокса.
        """
        return pd.Series(transformer.inverse_transform(data.to_frame()).flatten(), index=data.index)

    def diff(self, data: pd.Series, lag: int, drop: bool = True) -> pd.Series:
        """
        Возвращает дифференцированный ряд без Nan.
        """
        diff_data = data.diff(lag)
        if drop:
            diff_data = diff_data.dropna()
        return diff_data

    def diff_reverse(self, base: pd.Series, diff: pd.Series, lag: int) -> pd.Series:
        """
        Отменяет операцию дифференцирования c сохранением индексов diff.
        """
        lag_base = base.to_list()
        diff_list = diff.to_list()
        for i in range(len(diff)):
            lag_base.append(lag_base[i] + diff_list[i])
        lag_base = lag_base[lag::]
        return pd.Series(lag_base, index=diff.index)
