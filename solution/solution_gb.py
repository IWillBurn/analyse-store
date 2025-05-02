import pandas as pd
from matplotlib import pyplot as plt

from solution.analyse.analyser import Analyser
from solution.models.cat_boost.model import CatBoostModel
from solution.models.exponential_smoothing.model import ExponentialSmoothingModel
from solution.models.sarimax.model import SarimaxModel
from solution.models.unobsorved_components.model import UnobservedComponentsModel
from solution.preprocessing.preprocessor import Preprocessor
from solution.scoring.scorer import Scorer
from solution.transformation.transformer import Transformer
from solution.visualizing.visualizer import Visualizer


class SolutionGB:
    def __init__(self, rows: int, columns: int, periods: dict):
        self.preprocessor = Preprocessor(self)
        self.transformer = Transformer(self)
        self.visualizer = Visualizer(self)
        self.analyser = Analyser(self)
        self.scorer = Scorer(self)

        self.rows = rows
        self.columns = columns
        self.periods = periods
        self.max_period = max(self.periods.values())

        self.scoring_results_columns = ["mean", "median", "quan75", "quan95", "std", "no_stationary", "is_stationary",
                                        "no_autocorrelation",
                                        "is_homoscedasticity", "is_mean_zero", "is_normality"]

        self.items = None
        self.features = None
        self.datasets = None
        self.native_zeros = None
        self.yeojohnson_models = None

        self.working = None
        self.train = None
        self.test = None

        self.diff = None
        self.diff_lag = None
        self.diff_prefix = None

        self.diff_seasonal = None
        self.diff_seasonal_lag = None
        self.diff_seasonal_prefix = None

        self.cat_boost_models = None

        self.predicts = None
        self.clean_predict = None
        self.diff_predicts = None

        self.scoring_results = None

    def load(self, store: str, data: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> None:
        """
        Загружает данные в класс и собирает фичи
        """
        self.items, self.datasets, _ = self.preprocessor.feature_engineering_store(store, data, calendar, prices)

    def feature_engineering_lags(self, options: dict) -> None:
        """
        Добавляет фичи на основе лагов
        """
        self.items, self.datasets = self.preprocessor.feature_engineering_lags_store(self.items,
                                                                                     self.datasets,
                                                                                     self.max_period,
                                                                                     options)
        self.features = self.datasets[self.items[0]].columns.to_list()

    def group_by(self, by: str):
        """
        Группирует данные по by
        """
        self.working = {}

        for (item, data) in self.datasets.items():
            self.working[item] = self.preprocessor.group_by(data, by)

    def visualize_dataset(self):
        """
        Рисует графики изменения продаж, цены и изменения продаж с событиями по всем предметам магазина.
        """
        fig, ax = plt.subplots(self.rows * 3, self.columns, figsize=(80, 25))

        for ind, (item, data) in enumerate(self.working.items()):
            i_col = ind % self.columns
            i_row = ind // self.columns
            self.visualizer.visualize_dataset(data, item + f"({ind})",
                                              ax=[ax[i_row * 3, i_col],
                                                  ax[i_row * 3 + 1, i_col],
                                                  ax[i_row * 3 + 2, i_col]],
                                              show=False)
        plt.show()

    def split_train_test(self) -> None:
        """
        Разделяет выборку согласно periods на train и test.
        """
        self.train = {}
        self.test = {}

        for ind, (item, data) in enumerate(self.working.items()):
            self.train[item] = data.copy(deep=True).iloc[:-self.max_period:]
            self.test[item] = {}
            for period_name, period in self.periods.items():
                if -self.max_period + period == 0:
                    self.test[item][period_name] = data.copy(deep=True).iloc[-self.max_period::]
                else:
                    self.test[item][period_name] = data.copy(deep=True).iloc[
                                                   -self.max_period:-self.max_period + period:]

    def general_analyse(self) -> None:
        """
        Общий анализ всех временных рядов всех товаров.
        """
        for ind, (item, data) in enumerate(self.train.items()):
            self.analyser.analyse_time_series(data['cnt'], str(item) + f"({ind})")

    def clean_zeros(self, native_zeros: set) -> None:
        """
        Отчищает нули из данных
        """
        self.native_zeros = native_zeros
        for ind, (item, data) in enumerate(self.train.items()):
            if ind not in self.native_zeros:
                was_zeros, now_zeros = self.transformer.complete_zeroes(data)
                print(f'{item}({ind}) zeroes: {was_zeros} -> {now_zeros}')

    def fit(self, fields: list) -> None:
        """
        Обучает ExponentialSmoothingModel, UnobservedComponentsModel и SarimaxModel.
        """
        self.cat_boost_models = {}

        for ind, (item, data) in enumerate(self.train.items()):
            cat_boost = CatBoostModel(fields)
            cat_boost.fit(data.drop(['store_id', 'item_id'], axis=1), self.test[item]['week'].drop(['store_id', 'item_id'], axis=1))
            self.cat_boost_models[item] = cat_boost

    def predict(self) -> None:
        """
        Предсказывает через ExponentialSmoothingModel, UnobservedComponentsModel и SarimaxModel
        на все периоды по всем товарам.
        """

        self.predicts = {}
        self.clean_predict = {}

        for ind, (item, data) in enumerate(self.train.items()):
            self.predicts[item] = {}
            self.clean_predict[item] = {}
            for period_name, period in self.periods.items():
                self.predicts[item][period_name] = {}
                self.clean_predict[item][period_name] = {}

        print("predicting catboost models ...")

        for ind, (item, data) in enumerate(self.train.items()):
            for period_name, period in self.periods.items():
                index = pd.RangeIndex(start=data.index[-1] + 1, stop=data.index[-1] + period + 1)
                catboost_pred = self.cat_boost_models[item].predict(self.test[item][period_name].drop(['store_id', 'item_id'], axis=1), index=index)
                self.predicts[item][period_name][self.cat_boost_models[item].name] = catboost_pred
                self.clean_predict[item][period_name][self.cat_boost_models[item].name] = catboost_pred.copy(deep=True)

    def scoring_predicts(self) -> None:
        """
        Собирает метрики и выводит графики предсказаний по всем периодам, товарам и моделям.
        """
        self.scoring_results = {}

        for ind, (item, item_predicts) in enumerate(self.clean_predict.items()):
            self.scoring_results[item] = {}

        for key, value in self.periods.items():
            fig, ax = plt.subplots(self.rows, self.columns, figsize=(80, 25))

            for ind, (item, item_predicts) in enumerate(self.clean_predict.items()):
                self.scoring_results[item][key] = pd.DataFrame(data=[], columns=self.scoring_results_columns)
                models_predicts = {}
                for model_name, model_predicts in item_predicts[key].items():
                    models_predicts[model_name] = model_predicts[-value::]
                    _, _, short = self.scorer.score(self.clean_predict[item][key][model_name],
                                                    self.test[item][key]['cnt'])
                    short.index = [model_name]
                    self.scoring_results[item][key] = pd.concat([self.scoring_results[item][key], short],
                                                                ignore_index=False)
                i_col = ind % self.columns
                i_row = ind // self.columns
                self.visualizer.visualize_result(self.test[item][key]['cnt'], models_predicts,
                                                 ax=ax[i_row, i_col], show=False)
            print(f"Результаты на {key}: ")
            plt.show()

    def print_best(self) -> None:
        """
        Выводит лучшую модель для каждого периода и товара по медиане MAEs.
        """
        for key, value in self.periods.items():
            print(f"Результаты по {key}:")
            for ind, (item, item_predicts) in enumerate(self.clean_predict.items()):
                ind_min = self.scoring_results[item][key]["median"].idxmin()
                print(f"{item}({ind}) best: {ind_min} ({self.scoring_results[item][key]['median'][ind_min]})")
