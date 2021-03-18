from distribution import *
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
from scipy import stats


def draw_cdf(research_item, capacity, _plot):
    distribution_class = research_item['distribution_class']
    a = research_item['a']
    b = research_item['b']
    _plot.set_title(distribution_class.__name__ + ' n=' + str(capacity))
    sample = distribution_class.rvs(capacity)
    ecdf = ECDF(sample)
    x = np.linspace(a, b)
    y_ecdf = ecdf(x)
    y_cdf = distribution_class.cdf(x)
    _plot.step(x, y_ecdf)
    _plot.plot(x, y_cdf)
    plt.savefig('cdf' + distribution_class.__name__)


def draw_kde(research_item, capcity, axs):
    def kde(samples, param):
        n_kde = stats.gaussian_kde(samples, bw_method="silverman")
        n_kde.set_bandwidth(n_kde.factor * param)
        return n_kde

    bandwidth = [0.5, 1.0, 2.0]
    distribution_class = research_item['distribution_class']
    a = research_item['a']
    b = research_item['b']

    for bandwidth, _plot in zip(bandwidth, axs):
        _plot.set_title(distribution_class.__name__ + ' n=' + str(capcity))
        _plot.set_xlabel(f"h = h_n*{bandwidth}")
        sample = distribution_class.rvs(capcity)
        x = np.linspace(a, b)
        y = distribution_class.pdf(x)
        _plot.plot(x, y)

        cur_kde = kde(sample, bandwidth)
        y_kde = cur_kde.evaluate(x)
        _plot.plot(x, y_kde)
        _plot.set_ylim([0, 1])
        _plot.grid()

    plt.savefig("kde n = " + str(capacity) + " " + distribution_class.__name__)


if __name__ == '__main__':
    capacities = [20, 60, 100]
    research_items = [{'distribution_class': Normal, 'a': -4, 'b': 4},
                      {'distribution_class': Cauchy, 'a': -4, 'b': 4},
                      {'distribution_class': Laplace, 'a': -4, 'b': 4},
                      {'distribution_class': Poisson, 'a': 6, 'b': 14},
                      {'distribution_class': Uniform, 'a': -4, 'b': 4}]
    for item in research_items:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        for capacity, plot in zip(capacities, axs):
            draw_cdf(item, capacity, plot)

        for capacity in capacities:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            draw_kde(item, capacity, axs)
