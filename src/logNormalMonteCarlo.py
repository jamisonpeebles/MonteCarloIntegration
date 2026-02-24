import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

#np.random.seed(0)  # Set a seed for reproducibility

#pdf of the log-normal distribution
def f(x, mu, sigma):
    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x) - mu)**2 / (2 * sigma**2))

def monte_carlo_integration_mean_lognormal(f, a, b, n, mu, sigma):
    #generates n random numbers between a and b
    x = np.random.uniform(a, b, n)
    
    #evaluates the function at each of the random numbers
    fx = f(x, mu, sigma) * x
    
    #calculates the average value of the function at the random numbers
    avg_fx = np.mean(fx)
    
    #multiplies the average value by the length of the interval to get the estimate of the integral
    mean_estimate = (b - a) * avg_fx
    
    return mean_estimate

def plot_convergence_lognormal(f, a, b, max_n, mu, sigma):
    sample_sizes = np.unique(np.logspace(0, np.log10(max_n), 100).astype(int))
    estimates = []
    for n in sample_sizes:
        estimate = monte_carlo_integration_mean_lognormal(f, a, b, n, mu, sigma)
        estimates.append(estimate)
    estimates_by_n = pd.DataFrame({'n': sample_sizes, 'estimate': estimates})

    plt.figure(figsize=(10, 6))
    plt.plot(estimates_by_n['n'], estimates_by_n['estimate'], label='Monte Carlo Estimate')
    actual_mean = np.exp(mu + sigma**2 / 2)
    plt.axhline(actual_mean, color='red', linestyle='--', label='Actual Mean of Log-Normal')
    plt.xscale('log')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Estimated Integral')
    plt.title('Convergence of Monte Carlo Integration for Log-Normal Distribution')
    plt.legend()
    plt.grid()
    plt.show()

    return estimate

def plot_multi_sigma_lognormal(f, a, b, n, mu, min_sigma, max_sigma, sigma_step):

    # Generates estimates for different sigma values
    sigma_values = np.arange(min_sigma, max_sigma + sigma_step, sigma_step)
    estimates = []
    actual_means = []
    residuals = []
    for sigma in sigma_values:
        # Estimates the mean of the log-normal PDF by sampling using the Monte Carlo method
        estimate = monte_carlo_integration_mean_lognormal(f, a, b, n, mu, sigma)
        estimates.append(estimate)

        # Calculates the actual mean of the log-normal distribution using the known formula
        actual_mean = np.exp(mu + sigma**2 / 2)
        actual_means.append(actual_mean)
        if np.isinf(actual_mean):
            actual_mean = np.nan  # Handle case where actual mean is infinite

        # Calculates the error between the estimate and the actual mean
        error = np.abs(estimate - actual_mean)
        residuals.append(error)

    #stores them in a DataFrame
    estimates_by_sigma = pd.DataFrame({'sigma': sigma_values, 'estimate': estimates, 'actual_mean': actual_means, 'residual': residuals})

    plt.figure(figsize=(10, 6))
    # Plots the residuals against sigma values
    plt.plot(estimates_by_sigma['sigma'], estimates_by_sigma['residual'], label='Residuals')
    plt.yscale('log')
    plt.xlabel('Sigma Evaluated')
    plt.ylabel('Absolute Residual')
    plt.title('Convergence of Monte Carlo Integration for Log-Normal Distribution')
    plt.legend()
    plt.grid()
    plt.show()

    formatters = {col: '{:>12.3e}'.format for col in estimates_by_sigma.columns}
    formatters['sigma'] = '{:>12.6f}'.format  # different format for one column
    print(estimates_by_sigma.to_string(index=False, formatters=formatters))   

if __name__ == "__main__":
    a = 0.1  # lower limit of integration
    b = 10   # upper limit of integration
    n = 10**5  # number of random samples
    mu = 0     # mean of the underlying normal distribution
    sigma = 100    # standard deviation of the underlying normal distribution
    
    estimate = plot_multi_sigma_lognormal(f, a, b, n, mu, min_sigma=1, max_sigma=40, sigma_step=1)
