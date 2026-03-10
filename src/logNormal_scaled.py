import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import comb
import math

#np.random.seed(0)  # Set a seed for reproducibility

#pdf of the normal distribution
def normal_pdf(x, mu, sigma):
    x = np.asarray(x)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(- (x - mu)**2 / (2 * sigma**2))

#pdf of the binomial distribution
def binomial_pdf(k, n, p):
    k = np.asarray(k)
    n = np.asarray(n)
    p = np.asarray(p)
    
    pdf = (comb(n, k, exact=False)) * (p**k) * ((1-p)**(n-k))

    return pdf

#pdf of the logNormal distribution
def logNormal_pdf(x, mu, sigma):
    x = np.asarray(x)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    normal = normal_pdf(x, mu, sigma)
    logNormal = np.exp(normal)
    return logNormal

# pdf of the logBinomial distribution
def logBinomial_pdf(k, n, p):
    k = np.asarray(k)
    n = np.asarray(n)
    p = np.asarray(p)

    binom = binomial_pdf(k, n, p)
    logBinomial = np.exp(binom)
    return logBinomial

def plot_scaling_attempt(x, logNormal, logBinomial):

    plt.figure(figsize=(10, 6))
    plt.plot(x, logNormal, label='LogNormal Distribution')
    plt.plot(x, logBinomial, label='LogBinomial Distribution')
    plt.xlabel('x_values')
    plt.ylabel('pdf values')
    plt.legend()
    plt.grid()
    plt.show()


def sample_mean_logNormal(f, a, b, n, mu, sigma):
    #generates n random numbers between a and b
    x = np.random.uniform(a, b, n)
    
    #evaluates the function at each of the random numbers
    fx = f(x, mu, sigma) * x
    
    #calculates the average value of the function at the random numbers
    avg_fx = np.mean(fx)
    
    #multiplies the average value by the length of the interval to get the estimate of the integral
    sample_mean = (b - a) * avg_fx
    
    return sample_mean

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
    x = np.linspace(0.01, 1, 100)

    mu = 0.5     # mean of the underlying normal distribution
    sigma = 0.705336798983    # standard deviation of the underlying normal distribution

    n = 100
    p = 0.005

    logNormal = logNormal_pdf(x, mu, sigma)
    logBinomial = logBinomial_pdf(x, n, p)

    plot_scaling_attempt(x, logNormal, logBinomial)


    





