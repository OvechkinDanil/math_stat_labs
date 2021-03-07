from distribution import Distribution
import matplotlib.pyplot as plt
import numpy as np




def draw_boxplot(distribution, size):
    fig, ax = plt.subplots()
    ax.boxplot(list(map(lambda x: distribution(x), size)), vert=False)
    ax.set_title(distribution.__name__)
    ax.set_yticklabels(list(map(str, size)))
    plt.savefig(distribution.__name__)

def proportion_of_emissions(distribution, size_list):
    averaging = 1000
    result = []

    for size in size_list:
        data = []
        for i in range(averaging):
            array = np.array(distribution(size))
            x1 = np.quantile(array, 0.25) - 3 / 2 * (np.quantile(array, 0.75) - np.quantile(array, 0.25))
            x2 = np.quantile(array, 0.75) + 3 / 2 * (np.quantile(array, 0.75) - np.quantile(array, 0.25))
            data.append(len(list(filter(lambda x: x < x1 or x > x2, array))) / size)
        result.append((sum(data) / len(data)).__round__(4))
    return result

def print_emissions(distribution_name, size, emissions):
    latex_code = ""
    for i in range(len(size)):
        latex_code += f" {distribution_name} n = {size[i]} & {emissions[i]} \\\\ \\hline \n"
    return latex_code


if __name__ == '__main__':
    tabel_latex = f"\\begin{{tabular}}{{| c | c |}} \hline Sample & Share of emissions \\\\ \\hline"

    size = [20, 100]
    distributions = [Distribution.normal, Distribution.cauchy,
                     Distribution.laplace, Distribution.poisson, Distribution.uniform]

    for dist in distributions:
        draw_boxplot(dist, size)
        emission = proportion_of_emissions(dist, size)
        tabel_latex += print_emissions(dist.__name__, size, emission)

    tabel_latex += f" \\end{{tabular}}"

print(tabel_latex)



