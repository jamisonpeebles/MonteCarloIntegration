import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#np.random.seed(0)  # Set a seed for reproducibility

#defines a function such that f(x) = x^2
def f(x):
    return x**2

def monte_carlo_integration(f, a, b, n):
    #generates n random numbers between a and b
    x = np.random.uniform(a, b, n)
    
    #evaluates the function at each of the random numbers
    fx = f(x)
    
    #calculates the average value of the function at the random numbers
    avg_fx = np.mean(fx)
    
    #multiplies the average value by the length of the interval to get the estimate of the integral
    #works as scaling factor to account for the width of the interval
    integral_estimate = (b - a) * avg_fx
    
    return integral_estimate

def plot_convergence(f, a, b, max_n):
    sample_sizes = np.unique(np.logspace(0, np.log10(max_n), 100).astype(int))
    estimates = []
    for n in sample_sizes:
        estimate = monte_carlo_integration(f, a, b, n)
        estimates.append(estimate)
    estimates_by_n = pd.DataFrame({'n': sample_sizes, 'estimate': estimates})

    plt.figure(figsize=(10, 6))
    plt.plot(estimates_by_n['n'], estimates_by_n['estimate'], label='Monte Carlo Estimate')
    plt.axhline((b**3 - a**3) / 3, color='red', linestyle='--', label='Actual Integral (1/3)')
    plt.xscale('log')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Estimated Integral')
    plt.title('Convergence of Monte Carlo Integration')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    a = 0  # lower limit of integration
    b = 1  # upper limit of integration
    n = 1000000  # number of random samples
    
    plot_convergence(f, a, b, n)
    estimate = monte_carlo_integration(f, a, b, n)
    print(f"Estimated integral of f(x) from {a} to {b} is: {estimate}")
    print(f"Actual integral of f(x) from {a} to {b} is: {(b**3 - a**3) / 3}")  # Actual integral of x^2 from 0 to 1 is 1/3