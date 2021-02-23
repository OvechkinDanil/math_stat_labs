import matplotlib.pyplot as plt
import numpy as np
from distribution import Distribution
from density import Density


def plot_histogram(research_func, size, density_smoothness=50, bins_num=20):
    fig_num = len(size)
    fig, ax = plt.subplots(1, fig_num, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.4)

    distribution_name = research_func['name']

    plt.suptitle(distribution_name + " distribution")
    for i in range(fig_num):
        distribution = research_func['distribution'](size[i])
        density_x = np.linspace(min(distribution), max(distribution), density_smoothness)
        density_y = research_func['density'](density_x)

        ax[i].hist(distribution, density=True, histtype='bar', color='blue',
                   edgecolor='black', alpha=0.3, bins=bins_num)
        ax[i].plot(density_x, density_y, color='b', linewidth=1)
        ax[i].set_title(f'n = {size[i]}')
        ax[i].set_ylabel("density")

    plt.savefig(distribution_name + ".png")


if __name__ == '__main__':
    size = [10, 50, 1000]
    research_func = [
            {'name': 'Normal', 'distribution': Distribution.normal, 'density': Density.normal},
            {'name': 'Cauchy', 'distribution': Distribution.cauchy, 'density': Density.cauchy},
            {'name': 'Laplace', 'distribution': Distribution.laplace, 'density': Density.laplace},
            {'name': 'Poisson', 'distribution': Distribution.poisson, 'density': Density.poisson},
            {'name': 'Uniform', 'distribution': Distribution.uniform, 'density': Density.uniform}
        ]

    for each_func in research_func:
        plot_histogram(each_func, size)

    plt.show()