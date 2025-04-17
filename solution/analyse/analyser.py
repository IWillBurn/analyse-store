import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft
from solution.utils.time_series import time_series_info


class Analyser:
    def __init__(self, solution):
        self.solution = solution

    def draw_period_dynamic(self, data: pd.Series, period: int, ax: list = None, show: bool = True) -> None:
        """
        Рисует график изменения значение по элементам периода.
        """
        size = len(data) // period

        periods = []
        xs = []
        for i in range(period):
            periods.append(data[i::period][:size:])
            xs.append([j + i * (size + 2) for j in range(size)])

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(len(periods)):
            ax.plot(xs[i], periods[i], linewidth=4, label=f"Period {i + 1}", color="black")

        ax.set_xlabel("Периоды")
        ax.set_ylabel("Продажи")
        ax.set_title("Динамика продаж по периодам")
        ax.grid(True)
        plt.xticks([])
        if show:
            plt.show()

    def analyse_time_series(self, data: pd.Series, title: str = "Анализируемый ряд") -> None:
        """
        Выводит параметры временного ряда:
        Стационарность
        Гомоскедатичность
        Среднее 0
        Нормальность
        """
        print(f"Анализ ({title})")
        print(time_series_info(data))

    def show_fourier_decomposition(self, data: pd.Series, freq: float) -> None:
        """
        Рисует частотный график временного ряда.
        """
        values = fft.fft(data.to_numpy(), axis=0)
        freqs = fft.fftfreq(len(data), freq)
        real = 2.0 / len(data) * np.abs(values)

        plt.plot(freqs[0:len(data) // 2], real[0:len(data) // 2], label='Разложение Фурье')
        plt.xlabel('Частота')
        plt.ylabel('Амплитуда')
        plt.legend()
        plt.show()
