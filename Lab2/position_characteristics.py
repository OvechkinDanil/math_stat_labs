import numpy as np
import math


class PositionCharacteristics:

    @staticmethod
    def sample_mean(x):
        return sum(x) / len(x)

    @staticmethod
    def sample_median(x):
        n = len(x)
        l = n // 2
        if n % 2 == 0:
            return (x[l] + x[l + 1]) / 2
        else:
            return x[l + 1]

    @staticmethod
    def z_R(x):
        return (x[0] + x[len(x) - 1]) / 2

    @staticmethod
    def z_Q(x):
        return (np.quantile(x, 1 / 4) + np.quantile(x, 3 / 4)) / 2

    @staticmethod
    def truncated_mean(x):
        n = len(x)
        r = n // 4
        return sum(x[r: n - r - 1]) / (n - 2 * r)

    @staticmethod
    def dispersion(x):
        s_mean = PositionCharacteristics.sample_mean(x)
        n = len(x)
        s = 0
        for i in range(1, n):
            s += (x[i] - s_mean) ** 2

        return s / n

    @staticmethod
    def variance(x):
        x = np.array(x)
        mean = PositionCharacteristics.sample_mean(x)
        return PositionCharacteristics.sample_mean(x * x) - mean * mean

    @staticmethod
    def correct_digits(vrnc: float):
        return max(0, round(-math.log10(abs(vrnc))))
