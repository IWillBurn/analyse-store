import numpy as np
import pandas as pd
from scipy import fft


class FourierModel:

    def __init__(self, n_harm: int, d: float):
        self.n_harm = n_harm
        self.d = d
        self.train = None
        self.indexes = None
        self.values = None
        self.freq = None
        self.important = None

    def fit(self, train: pd.Series):
        self.train = train
        self.values = fft.fft(train)
        self.freq = fft.fftfreq(len(train), self.d)
        abs_values = np.abs(self.values[1:len(train) // 2])
        ind = np.argpartition(abs_values, -self.n_harm)[-self.n_harm:]
        self.important = np.sort(ind + 1)

    def summary(self):
        pass

    def predict(self, count: int, index: pd.Index) -> pd.Series:
        t = np.arange(len(self.train), len(self.train) + count)
        predict = np.zeros(len(t))

        for i in self.important:
            amplitude = 2 * np.abs(self.values[i]) / len(self.train)
            phase = np.angle(self.values[i])
            predict += amplitude * np.cos(2 * np.pi * self.freq[i] * t + phase)

        return pd.Series(predict, index=index)
