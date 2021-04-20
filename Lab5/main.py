import research

if __name__ == '__main__':
    capacity = [20, 60, 100]
    correlation_coefficients = [0.0, 0.5, 0.9]
    repetitions = 1000

    research.bivariate_normal_distribution(capacity, correlation_coefficients, repetitions)

    research.mixture_of_normal_distributions(capacity, repetitions)

    research.equiprobability_ellipse(capacity)