import numpy as np

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

if __name__ == "__main__":
    a = 0  # lower limit of integration
    b = 1  # upper limit of integration
    n = 1000000  # number of random samples
    
    estimate = monte_carlo_integration(f, a, b, n)
    print(f"Estimated integral of f(x) from {a} to {b} is: {estimate}")
    print(f"Actual integral of f(x) from {a} to {b} is: {(b**3 - a**3) / 3}")  # Actual integral of x^2 from 0 to 1 is 1/3