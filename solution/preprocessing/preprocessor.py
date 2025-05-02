import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, solution):
        self.solution = solution

    def feature_engineering_store(self,
                                  store: str,
                                  data: pd.DataFrame,
                                  calendar: pd.DataFrame,
                                  prices: pd.DataFrame) -> (list, dict, dict):
        """
        Применяет feature_engineering для всех предметов из магазина.
        """
        items = sorted(data[data['store_id'] == store]['item_id'].unique())

        datasets = {}
        results = {}

        for i in items:
            d = data[(data['store_id'] == store) & (data['item_id'] == i)]
            p = prices[(prices['store_id'] == store) & (prices['item_id'] == i)]
            dataset, result = self.feature_engineering(d, calendar, p)
            datasets[i] = dataset
            results[i] = result

        return items, datasets, results

    def feature_engineering_lags_store(self,
                                       items: list,
                                       datasets: dict,
                                       padding: int,
                                       options: dict) -> (list, dict):
        """
        Применяет feature_engineering_lags для всех предметов из магазина.
        """
        for i in items:
            datasets[i] = self.feature_engineering_lags(datasets[i], padding, options)

        return items, datasets

    def feature_engineering(self,
                            data: pd.DataFrame,
                            calendar: pd.DataFrame,
                            prices: pd.DataFrame) -> (pd.DataFrame, dict):
        """
        Формирует датасет из данных по продажам, календаря и стоимостям товаров.
        Добавляет категориальные фичи праздников.
        Добавляет ценовую фичу.
        Добавляет фичи abs_day, abs_week, abs_month для упрощения group_by.
        """
        data_calendar = pd.merge(data, calendar, on="date_id", how='inner')
        full_data = pd.merge(data_calendar, prices[['wm_yr_wk', 'sell_price']], on=["wm_yr_wk"],
                             how='inner')

        events_1 = set(full_data['event_name_1'].dropna().unique().tolist())
        events_2 = set(full_data['event_name_2'].dropna().unique().tolist())
        events = list(events_1.union(events_2))
        event_types_1 = set(full_data['event_type_1'].dropna().unique().tolist())
        event_types_2 = set(full_data['event_type_2'].dropna().unique().tolist())
        event_types = list(event_types_1.union(event_types_2))

        full_data['event_name_1'] = full_data['event_name_1'].fillna('-')
        full_data['event_name_2'] = full_data['event_name_2'].fillna('-')
        full_data['event_type_1'] = full_data['event_type_1'].fillna('-')
        full_data['event_type_2'] = full_data['event_type_2'].fillna('-')

        event_codes = []
        for event in events:
            event_code = 'EVENT_' + str(event)
            event_codes.append(event_code)
            full_data[event_code] = np.where(
                (full_data['event_name_1'] == event) | (full_data['event_name_2'] == event), 1, 0)

        event_type_codes = []
        for event_type in event_types:
            event_type_code = 'EVENTTYPE_' + str(event_type)
            event_type_codes.append(event_type_code)
            full_data[event_type_code] = np.where(
                (full_data['event_type_1'] == event_type) | (full_data['event_type_2'] == event_type), 1, 0)

        weekdays = set(full_data['weekday'].dropna().unique().tolist())
        weekdays_codes = []
        for weekday in weekdays:
            weekday_code = 'WEEKDAY_' + str(weekday)
            weekdays_codes.append(weekday_code)
            full_data[weekday_code] = np.where(full_data['weekday'] == weekday, 1, 0)

        full_data = full_data.drop('event_name_1', axis=1)
        full_data = full_data.drop('event_name_2', axis=1)
        full_data = full_data.drop('event_type_1', axis=1)
        full_data = full_data.drop('event_type_2', axis=1)
        full_data = full_data.drop('weekday', axis=1)
        full_data = full_data.drop('wm_yr_wk', axis=1)

        full_data = full_data.ffill()

        full_data['date'] = pd.to_datetime(full_data['date'])
        full_data = full_data.set_index('date')
        full_data = full_data.sort_index()

        start_date = min(full_data.index)
        end_date = min(full_data.index)
        expected_dates = pd.date_range(start=start_date, end=end_date)
        missing_dates = expected_dates.difference(full_data.index)

        missing_dates_count = len(missing_dates)

        if missing_dates_count != 0:
            print(f"SOME DATES ({len(missing_dates)}) ARE MISSED: {print(missing_dates)}, REINDEXING...")
            full_data = full_data.reindex(expected_dates)
            full_data = full_data.ffill()

        full_data['wday'] = 0
        for i in range(1, len(full_data)):
            full_data.loc[full_data.index[i], 'wday'] = ((full_data.loc[full_data.index[i - 1], 'wday'] + 1) % 7)

        full_data.index = pd.to_datetime(full_data.index)
        full_data = full_data.sort_index()

        full_data['day'] = full_data.index.day
        full_data['month'] = full_data.index.month
        full_data['year'] = full_data.index.year

        full_data['abs_day'] = pd.Series(range(0, len(full_data)), index=full_data.index)
        full_data['abs_month'] = (full_data['month'] + full_data['year'] * 12)
        full_data['abs_month'] = full_data['abs_month'] - min(full_data['abs_month'])

        full_data['abs_week'] = 0
        for i in range(1, len(full_data)):
            if full_data.loc[full_data.index[i], 'wday'] != 0:
                full_data.loc[full_data.index[i], 'abs_week'] = full_data.loc[full_data.index[i - 1], 'abs_week']
            else:
                full_data.loc[full_data.index[i], 'abs_week'] = full_data.loc[full_data.index[i - 1], 'abs_week'] + 1

        feature_engineering_result = {}

        feature_engineering_result['missing_dates_count'] = missing_dates_count
        feature_engineering_result['events'] = events
        feature_engineering_result['event_types'] = event_types
        feature_engineering_result['event_codes'] = event_codes
        feature_engineering_result['event_type_codes'] = event_type_codes

        return full_data, feature_engineering_result

    def feature_engineering_lags(self,
                                 data: pd.DataFrame,
                                 padding: int,
                                 options: dict) -> pd.DataFrame:
        """
        Добавляет фичи на основе предыдущих значений ряда (лаги и скользящие средние).
        """
        for i in range(options['lag_count']):
            data[f'LAG_{i + 1}'] = data['cnt'].shift(padding + i + 1).fillna(0)

        for i in range(options['seasonal_lag_count']):
            data[f'LAG_SEASONAL_{i + 1}'] = data['cnt'].shift(padding + (i + 1) * options['seasonal_lag']).fillna(0)

        for mean_size in options['mean_sizes']:
            data[f'ROLLING_MEAN_{mean_size}_LAG'] = data['cnt'].shift(padding + 1).rolling(window=mean_size, min_periods=1).mean()

        return data

    def feature_engineering_report(self, data: pd.DataFrame, result: dict) -> None:
        """
        Выводит отчёт по датасету.
        """
        nans = data['cnt'].isnull().sum()
        zeros = len(data[data['cnt'] == 0])
        columns = data.columns.tolist()

        print("Информация о датасете:")
        print(f"    Календарь:")
        print(
            f"        Дат отcуствовало: {result['missing_dates_count']} ({round(result['missing_dates_count'] / len(data) * 100, 3)}%)")
        print(f"        Уникальных событий: {len(result['events'])}")
        print(f"        Уникальных типов событий: {len(result['event_types'])}")
        print(f"    Данные:")
        print(f"        Nan таргетов: {nans} ({round(nans / len(data) * 100, 3)}%)")
        print(f"        Zero таргетов: {zeros} ({round(zeros / len(data) * 100, 3)}%)")
        print(f"        Даты: {min(data.index)} -> {max(data.index)}")
        print(
            f"        Дней: {len(data['abs_day'].unique())}; Недель: {len(data['abs_week'].unique())}; Месяцев: {len(data['abs_month'].unique())}")
        print(
            f"        Таргет: mean: {round(np.mean(data['cnt']), 3)}; median: {round(np.median(data['cnt']), 3)}; min: {round(np.min(data['cnt']), 3)}; max: {round(np.max(data['cnt']), 3)}")
        print(
            f"        Цена: mean: {round(np.mean(data['sell_price']), 3)}; median: {round(np.median(data['sell_price']), 3)}; min: {round(np.min(data['sell_price']), 3)}; max: {round(np.max(data['sell_price']), 3)}")
        print(f"    Фичи:")
        print(f"        Количество: {len(columns)}")
        print(f"        Фичи: {columns}")
        print(f"    События:")
        print(f"        По каждому событию:")
        for event in result['event_codes']:
            if event not in data.columns:
                continue
            indexes = data[event] != 0
            cnt = len(data[indexes])
            mult = data[indexes][event] * data[indexes]['cnt']
            avg = mult.sum() / cnt
            median = np.median(data[indexes]['cnt'])
            print(f"            {event}: cnt: {cnt}; avg: {round(avg, 3)}, median: {round(median, 3)}")
        print(f"        По каждому типу события:")
        for event_type in result['event_type_codes']:
            if event_type not in data.columns:
                continue
            indexes = data[event_type] != 0
            cnt = len(data[indexes])
            mult = data[indexes][event_type] * data[indexes]['cnt']
            avg = mult.sum() / cnt
            median = np.median(data[indexes]['cnt'])
            print(f"            {event_type}: cnt: {cnt}; avg: {round(avg, 3)}, median: {round(median, 3)}")

    def group_by(self, data: pd.DataFrame, by: str) -> pd.DataFrame:
        """
        Возвращает датасет сгрупированный по by ('abs_day', 'abs_month', 'abs_week').
        Цены усредняются; Id поддерживаются; Остальные поля складываются.
        """
        service = {'date_id', 'wday', 'month', 'year', 'day', 'abs_day', 'abs_month', 'abs_week'}
        service.remove(by)
        service = list(service)
        data = data.copy(deep=True)
        data = data.drop(columns=service)
        aggregation_methods = {col: 'sum' for col in data.columns if col != by}
        aggregation_methods.update({
            'item_id': 'first',
            'store_id': 'first',
            'sell_price': 'mean'
        })

        data = data.groupby(by).agg(aggregation_methods)
        return data
