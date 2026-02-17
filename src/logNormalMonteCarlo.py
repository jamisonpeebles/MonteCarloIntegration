import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

#np.random.seed(0)  # Set a seed for reproducibility

#pdf of the log-normal distribution
def f(x, mu, sigma):
    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x) - mu)**2 / (2 * sigma**2))

def monte_carlo_integration_lognormal(f, a, b, n, mu, sigma):
    #generates n random numbers between a and b
    x = np.random.uniform(a, b, n)
    
    #evaluates the function at each of the random numbers
    fx = f(x, mu, sigma)
    
    #calculates the average value of the function at the random numbers
    avg_fx = np.mean(fx)
    
    #multiplies the average value by the length of the interval to get the estimate of the integral
    integral_estimate = (b - a) * avg_fx
    
    return integral_estimate

def plot_convergence_lognormal(f, a, b, max_n, mu, sigma):
    sample_sizes = np.unique(np.logspace(0, np.log10(max_n), 100).astype(int))
    estimates = []
    for n in sample_sizes:
        estimate = monte_carlo_integration_lognormal(f, a, b, n, mu, sigma)
        estimates.append(estimate)
    estimates_by_n = pd.DataFrame({'n': sample_sizes, 'estimate': estimates})

    plt.figure(figsize=(10, 6))
    plt.plot(estimates_by_n['n'], estimates_by_n['estimate'], label='Monte Carlo Estimate')
    actual_integral = norm.cdf(np.log(b), loc=mu, scale=sigma) - norm.cdf(np.log(a), loc=mu, scale=sigma)
    plt.axhline(actual_integral, color='red', linestyle='--', label='Actual Integral')
    plt.xscale('log')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Estimated Integral')
    plt.title('Convergence of Monte Carlo Integration for Log-Normal Distribution')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    a = 0.1  # lower limit of integration
    b = 10   # upper limit of integration
    n = 1000000  # number of random samples
    mu = 0     # mean of the underlying normal distribution
    sigma = 1  # standard deviation of the underlying normal distribution
    
    plot_convergence_lognormal(f, a, b, n, mu, sigma)
    estimate = monte_carlo_integration_lognormal(f, a, b, n, mu, sigma)
    actual_integral = norm.cdf(np.log(b), loc=mu, scale=sigma) - norm.cdf(np.log(a), loc=mu, scale=sigma)
    
    print(f"Estimated integral of log-normal PDF from {a} to {b} is: {estimate}")
    print(f"Actual integral of log-normal PDF from {a} to {b} is: {actual_integral}")
    