import os

import joblib
import pandas as pd
from matplotlib import pyplot as plt

from solution.analyse.analyser import Analyser
from solution.models.catboost.model import CatBoostModel
from solution.models.exponential_smoothing.model import ExponentialSmoothingModel
from solution.models.sarimax.model import SarimaxModel
from solution.models.unobsorved_components.model import UnobservedComponentsModel
from solution.preprocessing.preprocessor import Preprocessor
from solution.scoring.scorer import Scorer
from solution.transformation.transformer import Transformer
from solution.visualizing.visualizer import Visualizer


class Solution:
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
        self.datasets = None
        self.native_zeros = None
        self.yeojohnson_models = None

        self.working = None
        self.train = None
        self.test = None

        self.stable = None
        self.diff = None
        self.diff_lag = None
        self.diff_prefix = None

        self.diff_seasonal = None
        self.diff_seasonal_lag = None
        self.diff_seasonal_prefix = None

        self.exp_smoothing_models = None
        self.unobserved_components_models = None
        self.sarimax_models = None
        self.catboost_models = None

        self.predicts = None
        self.clean_predict = None
        self.diff_predicts = None
        self.stable_predicts = None

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

    def load_test(self, start: int) -> None:
        """
        Загружает сэмпл из данных как test.
        """
        self.test = {}
        for ind, (item, data) in enumerate(self.working.items()):
            self.test[item] = {}
            for period_name, period in self.periods.items():
                self.test[item][period_name] = data.copy(deep=True).iloc[start:start + period:]

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

    def variance_stabilization(self) -> None:
        """
        Выравнивает дисперсию
        """
        self.stable = {}
        self.yeojohnson_models = {}
        for ind, (item, data) in enumerate(self.train.items()):
            yeojohnson_model, yeojohnsoned = self.transformer.yeojohnson(data['cnt'])
            self.yeojohnson_models[item] = yeojohnson_model

            data_cp = data.copy(deep=True)
            data_cp['cnt'] = yeojohnsoned
            self.stable[item] = data_cp

    def do_diff(self, lag: int) -> None:
        """
        Дифференцирует временные ряды всех товаров.
        """
        self.diff = {}
        self.diff_lag = lag
        self.diff_prefix = {}
        for ind, (item, data) in enumerate(self.stable.items()):
            diff_data = self.transformer.diff(data['cnt'], lag)
            self.analyser.analyse_time_series(diff_data, title=item + f"({ind})")

            data_cp = data.copy(deep=True)
            data_cp["cnt"] = self.transformer.diff(data['cnt'], lag, drop=False)
            self.diff[item] = data_cp.iloc[lag::]
            self.diff_prefix[item] = data['cnt'][:lag:]

    def do_diff_seasonal(self, lag: int) -> None:
        """
        Дифференцирует временные ряды всех товаров сезонно.
        """
        self.diff_seasonal = {}
        self.diff_seasonal_lag = lag
        self.diff_seasonal_prefix = {}
        for ind, (item, data) in enumerate(self.stable.items()):
            diff_data = self.transformer.diff(data['cnt'], lag)
            self.analyser.analyse_time_series(diff_data, title=item + f"({ind})")

            data_cp = data.copy(deep=True)
            data_cp["cnt"] = self.transformer.diff(data['cnt'], lag, drop=False)
            self.diff_seasonal[item] = data_cp.iloc[lag::]
            self.diff_seasonal_prefix[item] = data['cnt'][:lag:]

    def visualize_acf_pacf(self, lag: int) -> None:
        """
        Рисует графики функций автокорреляций и частичных автокорреляций для дифференцированных данных.
        """
        fig, ax = plt.subplots(self.rows * 2, self.columns, figsize=(80, 25))

        for ind, (item, data) in enumerate(self.stable.items()):
            diff_data = self.transformer.diff(data['cnt'], lag)

            i_col = ind % self.columns
            i_row = ind // self.columns
            self.visualizer.visualize_acf_pacf(diff_data, item + f"({ind})",
                                               ax=[ax[i_row * 2, i_col], ax[i_row * 2 + 1, i_col]], show=False)

        plt.show()

    def fit(self,
            exponential_smoothing_space: list,
            unobserved_components_space: list,
            sarimax_space: dict,
            catboost_fields: list) -> None:
        """
        Обучает ExponentialSmoothingModel, UnobservedComponentsModel, SarimaxModel и CatBoost.
        """
        self.exp_smoothing_models = {}
        self.unobserved_components_models = {}
        self.sarimax_models = {}
        self.catboost_models = {}

        if exponential_smoothing_space is not None:
            print("fitting exponential smoothing models ...")
            for ind, (item, data) in enumerate(self.diff.items()):
                exp_smoothing = ExponentialSmoothingModel(exponential_smoothing_space)
                exp_smoothing.fit(data['cnt'])
                self.exp_smoothing_models[item] = exp_smoothing

        if unobserved_components_space is not None:
            print("fitting unobserved components models ...")
            for ind, (item, data) in enumerate(self.diff.items()):
                unobserved_components = UnobservedComponentsModel(unobserved_components_space)
                unobserved_components.fit(data, None)
                self.unobserved_components_models[item] = unobserved_components

        if sarimax_space is not None:
            print("fitting sarimax models ...")
            for ind, (item, data) in enumerate(self.stable.items()):
                arima = SarimaxModel(auto_p=sarimax_space["p"],
                                     auto_q=sarimax_space["q"],
                                     auto_P=sarimax_space["P"],
                                     auto_Q=sarimax_space["Q"],
                                     auto_d=sarimax_space["d"],
                                     auto_D=sarimax_space["D"],
                                     auto_seasonal_period=sarimax_space["seasonal_period"])
                arima.fit_auto(data['cnt'], None)
                self.sarimax_models[item] = arima

        if catboost_fields is not None:
            print("fitting catboost models ...")
            for ind, (item, data) in enumerate(self.train.items()):
                catboost = CatBoostModel(catboost_fields)
                catboost.fit(data.drop(['store_id', 'item_id'], axis=1),
                             self.test[item]['week'].drop(['store_id', 'item_id'], axis=1))
                self.catboost_models[item] = catboost

    def predict(self) -> None:
        """
        Предсказывает через ExponentialSmoothingModel, UnobservedComponentsModel, SarimaxModel и CatBoost
        на все периоды по всем товарам.
        """

        self.predicts = {}
        self.clean_predict = {}
        self.diff_predicts = set()
        self.stable_predicts = set()

        for item in self.items:
            self.predicts[item] = {}
            self.clean_predict[item] = {}
            for period_name, period in self.periods.items():
                self.predicts[item][period_name] = {}
                self.clean_predict[item][period_name] = {}

        if len(self.exp_smoothing_models) > 0:
            print("predicting exponential smoothing models ...")

            for ind, (item, data) in enumerate(self.test.items()):
                for period_name, period in self.periods.items():
                    index = pd.RangeIndex(data[period_name].index[0], stop=data[period_name].index[0] + period)
                    exp_smoothing_preds = self.exp_smoothing_models[item].predict(period, index=index)
                    for model_name, pred in exp_smoothing_preds.items():
                        self.predicts[item][period_name][model_name] = pred
                        self.clean_predict[item][period_name][model_name] = pred.copy(deep=True)
                        self.diff_predicts.add(model_name)
                        self.stable_predicts.add(model_name)

        if len(self.unobserved_components_models) > 0:
            print("predicting unobserved components models ...")

            for ind, (item, data) in enumerate(self.test.items()):
                for period_name, period in self.periods.items():
                    index = pd.RangeIndex(data[period_name].index[0], stop=data[period_name].index[0] + period)
                    unobserved_components_preds = self.unobserved_components_models[item].predict(period, None,
                                                                                                  index=index)
                    for model_name, pred in unobserved_components_preds.items():
                        self.predicts[item][period_name][model_name] = pred
                        self.clean_predict[item][period_name][model_name] = pred.copy(deep=True)
                        self.diff_predicts.add(model_name)
                        self.stable_predicts.add(model_name)

        if len(self.sarimax_models) > 0:
            print("predicting sarimax models ...")

            for ind, (item, data) in enumerate(self.test.items()):
                for period_name, period in self.periods.items():
                    index = pd.RangeIndex(start=data[period_name].index[0], stop=data[period_name].index[0] + period)
                    arima_pred = self.sarimax_models[item].predict_auto(period, None, index=index)
                    self.predicts[item][period_name][self.sarimax_models[item].name] = arima_pred
                    self.clean_predict[item][period_name][self.sarimax_models[item].name] = arima_pred.copy(deep=True)
                    self.stable_predicts.add(self.sarimax_models[item].name)

        if len(self.catboost_models) > 0:
            print("predicting catboost models ...")

            for ind, (item, data) in enumerate(self.test.items()):
                for period_name, period in self.periods.items():
                    index = pd.RangeIndex(start=data[period_name].index[0], stop=data[period_name].index[0] + period)
                    catboost_pred = self.catboost_models[item].predict(
                        self.test[item][period_name].drop(['store_id', 'item_id'], axis=1), index=index)
                    self.predicts[item][period_name][self.catboost_models[item].name] = catboost_pred
                    self.clean_predict[item][period_name][self.catboost_models[item].name] = catboost_pred.copy(
                        deep=True)

    def predicts_reverse_diff(self) -> None:
        """
        Отменяет дифференцирование для всех предсказаний, которым это требуется.
        """
        for item, item_predicts in self.clean_predict.items():
            for period_name, periods_predicts in item_predicts.items():
                for model_name, models_pred in periods_predicts.items():
                    if model_name in self.diff_predicts:
                        dediff = self.transformer.diff_reverse(self.diff_prefix[item], models_pred, self.diff_lag)
                        self.clean_predict[item][period_name][model_name] = dediff

    def predicts_reverse_yeojohnson(self) -> None:
        """
        Отменяет преобразование Йео-Джонсона
        """
        for item, item_predicts in self.clean_predict.items():
            for period_name, periods_predicts in item_predicts.items():
                for model_name, models_pred in periods_predicts.items():
                    if model_name in self.stable_predicts:
                        deyeojohnsoned = self.transformer.yeojohnson_reverse(self.yeojohnson_models[item], models_pred)
                        deyeojohnsoned_clean = deyeojohnsoned.fillna(
                            (deyeojohnsoned.shift(-1) + deyeojohnsoned.shift(1)) / 2).ffill().bfill()
                        self.clean_predict[item][period_name][model_name] = deyeojohnsoned_clean

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

    def save_models(self,
                    path: str,
                    exponential_smoothing: bool = False,
                    unobserved_components: bool = False,
                    sarimax: bool = False,
                    catboost: bool = False) -> None:
        """
        Сохраняет выбранные модели
        """
        for item in self.items:
            if exponential_smoothing:
                joblib.dump(self.exp_smoothing_models[item], path + f'exponential_smoothing_{item}.joblib')
            if unobserved_components:
                joblib.dump(self.unobserved_components_models[item], path + f'unobserved_components_{item}.joblib')
            if sarimax:
                joblib.dump(self.sarimax_models[item], path + f'sarimax_{item}.joblib')
            if catboost:
                joblib.dump(self.catboost_models[item], path + f'catboost_{item}.joblib')

    def load_models(self, path: str):
        self.exp_smoothing_models = {}
        self.unobserved_components_models = {}
        self.sarimax_models = {}
        self.catboost_models = {}

        types = [
            "exponential_smoothing",
            "unobserved_components",
            "sarimax",
            "catboost",
        ]

        skip = False

        for model_name in types:
            for item in self.items:
                full_path = path + f'{model_name}_{item}.joblib'
                if os.path.exists(full_path):
                    model = joblib.load(full_path)
                    if model_name == "exponential_smoothing":
                        self.exp_smoothing_models[item] = model
                    elif model_name == "unobserved_components":
                        self.unobserved_components_models[item] = model
                    elif model_name == "sarimax":
                        self.sarimax_models[item] = model
                    elif model_name == "catboost":
                        self.catboost_models[item] = model
                else:
                    print(f'model {full_path} not found, {model_name} skipping...')
                    if model_name == "exponential_smoothing":
                        self.exp_smoothing_models.clear()
                    elif model_name == "unobserved_components":
                        self.unobserved_components_models.clear()
                    elif model_name == "sarimax":
                        self.sarimax_models.clear()
                    elif model_name == "catboost":
                        self.catboost_models.clear()
                    skip = True
                    break
            if skip:
                skip = False
            else:
                print(f'model {model_name} loaded...')
