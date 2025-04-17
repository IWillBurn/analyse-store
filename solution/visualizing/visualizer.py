import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots


class Visualizer:
    def __init__(self, solution):
        self.solution = solution

    def visualize_dataset(self, data: pd.DataFrame, title: str, ax: list = None, show: bool = True) -> None:
        """
        Рисует графики изменения продаж, цены и изменения продаж с событиями.
        """
        if ax is None:
            fig, ax = plt.subplots(3, 1, figsize=(60, 25))

        ax[0].plot(data['cnt'], label='cnt')
        ax[0].set_title(title)
        ax[0].set_ylabel('Продажи')

        ax[1].plot(data['sell_price'], label='price')
        ax[1].set_title(f'Стоимость ({title})')
        ax[1].set_xlabel('Даты')
        ax[1].set_ylabel('Стоимость')

        events = [c for c in data.columns if c[:10:] == "EVENTTYPE_"]
        ax[2].plot(data['cnt'], alpha=0.5, c='black', label='data')
        for ind, event in enumerate(events):
            x = data[data[event] != 0].index
            y = data.loc[x]['cnt']
            ax[2].plot(x, y, 'o', alpha=1, label=event)
        ax[2].legend()
        ax[2].set_title(f'Отметки событий {title}')
        ax[2].set_xlabel('Даты')
        ax[2].set_ylabel('Продажи')
        if show:
            plt.show()

    def visualize_acf_pacf(self, data: pd.Series, title: str, ax: list = None, show: bool = True) -> None:
        """
        Рисует графики функций автокорреляций и частичных автокорреляций.
        """
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(80, 25))
        tsaplots.plot_acf(data, ax=ax[0], title=f"Автокореляции ({title})")
        tsaplots.plot_pacf(data, ax=ax[1], title=f"Частичные автокореляции ({title})")
        if show:
            plt.show()

    def visualize_result(self, test, predicts, ax: plt.Axes = None, show: bool = True) -> None:
        """
        Рисует графики предиктов моделей и верных ответов на одном графике.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(40, 15))
        for (name, predict) in predicts.items():
            ax.plot(predict[-len(test)::], alpha=0.8, label=name)

        ax.plot(test, alpha=0.8, c='black', label='fact')
        ax.legend()

        if show:
            plt.show()
