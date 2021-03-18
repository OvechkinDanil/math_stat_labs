import numpy as np
import scipy.stats as stats
import math
from scipy.special import factorial


class Normal:
    @staticmethod
    def rvs(capacity):
        return np.random.normal(0, 1, capacity)

    @staticmethod
    def cdf(x):
        return stats.norm(0, 1).cdf(x)

    @staticmethod
    def pdf(x):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2)

class Cauchy:
    @staticmethod
    def rvs(capacity):
        return np.random.standard_cauchy(capacity)

    @staticmethod
    def cdf(x):
        return stats.cauchy(loc=0, scale=1).cdf(x)

    @staticmethod
    def pdf(x):
        return 1 / np.sqrt(2) * np.exp(-np.sqrt(2) * np.abs(x))

class Laplace:
    @staticmethod
    def rvs(capacity):
        return np.random.laplace(0, np.sqrt(2), capacity)

    @staticmethod
    def cdf(x):
        return stats.laplace(loc=0, scale=1 / math.sqrt(2)).cdf(x)

    @staticmethod
    def pdf(x):
        return 1 / np.sqrt(2) * np.exp(-np.sqrt(2) * np.abs(x))

class Poisson:
    @staticmethod
    def rvs(capacity):
        return np.random.poisson(10, capacity)

    @staticmethod
    def cdf(x):
        return stats.poisson(10).cdf(x)

    @staticmethod
    def pdf(x):
        return np.power(10, x) / factorial(x) * np.exp(-10)

class Uniform:
    @staticmethod
    def rvs(capacity):
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), capacity)

    @staticmethod
    def cdf(x):
        return stats.uniform(-math.sqrt(3), 2 * math.sqrt(3)).cdf(x)

    @staticmethod
    def pdf(x):
        pdf = []
        for each_x in x:
            if math.fabs(each_x) <= math.sqrt(3):
                pdf.append(1 / (2 * math.sqrt(3)))
            else:
                pdf.append(0)
        return pdf
