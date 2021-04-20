import scipy.stats as stats
import numpy as np

class CorrelationCoefficients:

    @staticmethod
    def pearson(sample):
        x = sample[:, 0]
        y = sample[:, 1]
        return stats.pearsonr(x, y)[0]
    
    @staticmethod
    def spearman(sample):
        x = sample[:, 0]
        y = sample[:, 1]
        return stats.spearmanr(x, y)[0]

    @staticmethod
    def quadrant(sample):
        x = sample[:, 0]
        y = sample[:, 1]

        x = x - np.median(x)
        y = y - np.median(y)

        n = np.zeros(4)

        for i in range(len(x)):
            if x[i] > 0 and y[i] >= 0:
                n[0] += 1
            elif x[i] <= 0 and y[i] > 0:
                n[1] += 1
            elif x[i] < 0 and y[i] <= 0:
                n[2] += 1
            elif x[i] > 0 and y[i] < 0:
                n[3] += 1

        return ((n[0] + n[2]) - (n[1] + n[3])) / len(x)
    


