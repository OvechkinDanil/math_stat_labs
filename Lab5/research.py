import scipy.stats as stats
import numpy as np
import statistics
import matplotlib.pyplot as plt
from ellipse import Ellipse as EllipseClass
from correlation_coefficients import CorrelationCoefficients as cc

def bivariate_normal_distribution(capacities, correlation_coefficients, repeat):

    for capacity in capacities:

        tabel_in_latex = f"\\begin{{tabular}}{{| c | c | c | c |}} \hline \n"

        for cor_cov in correlation_coefficients:
            tabel_in_latex += f" p = {cor_cov} & $r$ & $r_{{S}}$ & $r_{{Q}}$ \\\\ \\hline \n"

            data = []
            for _ in range(repeat):
                sample = stats.multivariate_normal.rvs([0, 0], [[1, cor_cov], [cor_cov, 1]], capacity)
                data += [[c_cor(sample) for c_cor in [cc.pearson, cc.spearman, cc.quadrant]]]
            data = np.array(data).T
            res = [list(map(lambda x: x.__round__(3), [
                np.mean(data[i]),
                np.mean(list(map(lambda x: x ** 2, data[i]))),
                statistics.variance(data[i])])) for i in range(3)]

            for cor_coeff_characteristics in range(len(res[0])):
                if cor_coeff_characteristics == 0:
                    tabel_in_latex += f" $E(z)$"
                elif cor_coeff_characteristics == 1:
                    tabel_in_latex += f" $E(z^2)$"
                elif cor_coeff_characteristics == 2:
                    tabel_in_latex += f" $D(z)$"

                for cor_coeff in res:
                    tabel_in_latex += f" & {cor_coeff[cor_coeff_characteristics]}"

                tabel_in_latex += f" \\\\ \\hline \n"

        tabel_in_latex += f" \\end{{tabular}} \n"
        print(tabel_in_latex)


def mixture_of_normal_distributions(capacities, repetitions):

    tabel_in_latex = f"\\begin{{tabular}}{{| c | c | c | c |}} \hline \n"

    for capacity in capacities:
        data = []
        for _ in range(repetitions):
            sample = 0.9 * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], capacity) + \
                     0.1 * stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], capacity)
            data += [[c_cor(sample) for c_cor in [cc.pearson, cc.spearman, cc.quadrant]]]
        data = np.array(data)
        res = [list(map(lambda x: x.__round__(3), [
            np.mean(data[i]),
            np.mean(list(map(lambda x: x ** 2, data[i]))),
            statistics.variance(data[i])])) for i in range(3)]

        tabel_in_latex += f" n = {capacity} & $r$ & $r_{{S}}$ & $r_{{Q}}$ \\\\ \\hline \n"

        for cor_coeff_characteristics in range(len(res[0])):
            if cor_coeff_characteristics == 0:
                tabel_in_latex += f" $E(z)$"
            elif cor_coeff_characteristics == 1:
                tabel_in_latex += f" $E(z^2)$"
            elif cor_coeff_characteristics == 2:
                tabel_in_latex += f" $D(z)$"

            for cor_coeff in res:
                tabel_in_latex += f" & {cor_coeff[cor_coeff_characteristics]}"

            tabel_in_latex += f" \\\\ \\hline \n"

    tabel_in_latex += f" \\end{{tabular}} \n"

    print(tabel_in_latex)

def equiprobability_ellipse(capacities):
    for capacity in capacities:
        _, sp = plt.subplots(1, 3, figsize=(16, 6))
        for cor_cov, subplot in zip([0, 0.5, 0.9], sp):
            sample = stats.multivariate_normal.rvs([0, 0], [[1, cor_cov], [cor_cov, 1]], capacity)

            x = sample[:, 0]
            y = sample[:, 1]

            ellipse = EllipseClass(0, 0,
                              1, 1, cor_cov)

            subplot.scatter(x, y)

            x = np.linspace(min(x) - 2, max(x) + 2, 100)
            y = np.linspace(min(y) - 2, max(y) + 2, 100)
            x, y = np.meshgrid(x, y)
            z = ellipse.z(x, y)
            t = ellipse.rad2(sample)
            subplot.contour(x, y, z, [ellipse.rad2(sample)])

            title = f"n = {capacity} rho = {cor_cov}"

            subplot.set_title(title)
            subplot.set_xlabel("X")
            subplot.set_ylabel("Y")

        plt.savefig(f"{capacity}" + f".png")




