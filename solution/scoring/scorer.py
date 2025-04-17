import numpy as np
import pandas as pd
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError, MeanAbsoluteError

from solution.utils.time_series import time_series_info


class Scorer:
    def __init__(self, solution):
        self.solution = solution
        pass

    def score(self, predict: pd.Series, true: pd.Series) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Считает оценку остатков и метрики предикта.
        """
        resids = self.score_resid(true, predict)
        metrics = self.score_metrics(true, predict)

        columns = ["mean", "median", "quan75", "quan95", "std", "no_stationary", "is_stationary", "no_autocorrelation",
                   "is_homoscedasticity", "is_mean_zero", "is_normality"]
        vals = []

        for index, row in metrics.iterrows():
            if index == "maes":
                for value in row:
                    vals.append(value)

        for index, row in resids[["val", "opinion"]].iloc[:-1].iterrows():
            sum = ""
            for value in row:
                sum += str(value)
            vals.append(sum)

        return resids, metrics, pd.DataFrame([vals], columns=columns)

    def calc_wape(self, predict: pd.Series, true: pd.Series) -> pd.Series:
        """
        Рассчитывает значение wape по индексам.
        """
        wapes = []
        abs_sum = 0
        delta_sum = 0
        p = predict.to_list()
        t = true.to_list()
        for p_e, t_e in zip(p, t):
            abs_sum += abs(t_e)
            delta_sum += abs(p_e - t_e)
            if abs_sum != 0:
                wapes.append(delta_sum / abs_sum)
            else:
                wapes.append(np.nan)
        return pd.Series(wapes, index=predict.index)


    def score_metrics(self, predict: pd.Series, true: pd.Series) -> pd.DataFrame:
        """
        Считает maes, mape, wape для предикта.
        """
        mapes = MeanAbsolutePercentageError().evaluate_by_index(true, predict) * 100
        maes = MeanAbsoluteError().evaluate_by_index(true, predict)
        wapes = self.calc_wape(true, predict)

        def quan_75(y, quan=0.75):
            return y.quantile(quan)

        def quan_95(y, quan=0.95):
            return y.quantile(quan)

        stats_list = ['mean', 'median', quan_75, quan_95, 'std']
        res = pd.DataFrame([maes.agg(stats_list), mapes.agg(stats_list), wapes.agg(stats_list)],
                           index=['maes', 'mapes', 'wapes'])

        return res

    def score_resid(self, predict: pd.Series, true: pd.Series) -> pd.DataFrame:
        """
        Оценивает остаток на стационарность, гетероскопичность, автокореляцию, нормальность.
        """
        resid = predict - true
        return time_series_info(resid)
