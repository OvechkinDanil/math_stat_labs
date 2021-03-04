from distribution import Distribution
from position_characteristics import PositionCharacteristics as pc
import numpy as np

size_array = [10, 100, 1000]
distributions = [Distribution.normal, Distribution.uniform, Distribution.poisson, Distribution.laplace, Distribution.cauchy]
position_characteristics = [pc.sample_mean, pc.sample_median, pc.z_R, pc.z_Q,  pc.truncated_mean]


if __name__ == '__main__':
    for dist in distributions:
        latex_code = f"\\begin{{tabular}}{{|c | c | c | c | c | c|}} \n \hline \multicolumn{{6}}{{|c|}}{{{dist.__name__}}} \\\\ \n"
        latex_code += f" \\hline & $\\bar{{x}}$ & $medx$ & $z_R$ & $z_Q$ & $z_{{tr}}$ \n"
        for size in size_array:
            latex_code += f" \\\\ \\hline $n={size}$ & & & & & \\\\ \n"
            mean = []
            variance = []

            for pos_char in position_characteristics:
                data = [pos_char(sorted(dist(size))) for i in range(1000)]
                mean.append(round(pc.sample_mean(data), 10))
                variance.append(pc.variance(data))

            latex_code += f" \\hline $E(z)$ \n"
            for elem in mean:
                latex_code += f" &{elem}"

            latex_code += f" \\\\ \n"
            latex_code += f" \\hline $D(z)$ \n"

            for elem in variance:
                latex_code += f" &{round(elem, 6)}"

            print("\n")

            latex_code += f" \\\\ \n"
            latex_code += f"\\hline $\hat{{E}}(z)$ \n"

            for i in range(len(mean)):
                latex_code += f"&{round(mean[i], pc.correct_digits(np.sqrt(variance[i])))}"

            latex_code += f" \\\\ \n"
            latex_code += f"\\hline $E-\sqrt{{D(z)}}$ \n"

            for i in range(len(mean)):
                latex_code += f"&{round(mean[i] - np.sqrt(variance[i]), 6)}"

            latex_code += f" \\\\ \n"
            latex_code += f"\\hline $E+\sqrt{{D(z)}}$ \n"

            for i in range(len(mean)):
                latex_code += f"&{round(mean[i] + np.sqrt(variance[i]), 6)}"

            print("\n")
        latex_code += f" \\\\ \\hline \n \end{{tabular}}"
        print(latex_code)

