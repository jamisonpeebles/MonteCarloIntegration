import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binom, norm

def plot_binom_normal(n, p):
    mean = n * p
    std_dev = np.sqrt(n * p * (1 - p))

    # Range of x values (focus on where the mass is)
    x_min = int(max(0, mean - 4 * std_dev))
    x_max = int(min(n, mean + 4 * std_dev))
    k = np.arange(x_min, x_max + 1)

    # Binomial PMF
    binom_pmf = binom.pmf(k, n, p)

    # Normal PDF over a continuous range
    x_cont = np.linspace(x_min, x_max, 300)
    normal_pdf = norm.pdf(x_cont, loc=mean, scale=std_dev)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(k, binom_pmf, color='blue', alpha=0.6, label='Binomial PMF')
    ax.plot(x_cont, normal_pdf, 'r-', linewidth=2, label='Normal PDF')
    ax.set_title(f'Binomial(n={n}, p={p}) vs Normal(μ={mean}, σ={std_dev:.2f})')
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_logBinom_logNormal(n, p):
    mean = n * p
    std_dev = np.sqrt(n * p * (1 - p))

    # Range of x values (focus on where the mass is)
    x_min = int(max(0, mean - 4 * std_dev))
    x_max = int(min(n, mean + 4 * std_dev))
    k = np.arange(x_min, x_max + 1)

    # exp(Binomial): discrete values at exp(k) with binomial probabilities
    exp_k = np.exp(k)
    binom_pmf = binom.pmf(k, n, p)

    # LogNormal PDF on a log-spaced continuous range
    from scipy.stats import lognorm
    exp_x_cont = np.logspace(np.log10(exp_k[0] * 0.5), np.log10(exp_k[-1] * 2), 500)
    lognorm_pdf = lognorm.pdf(exp_x_cont, s=std_dev, scale=np.exp(mean))

    # Scale PMF to density by dividing by gap width between consecutive exp(k)
    gaps = np.diff(exp_k)
    gaps = np.append(gaps, gaps[-1])
    binom_density = binom_pmf / gaps

    print(pd.DataFrame({'exp_k': exp_k, 'binom_density': binom_density, 'lognorm_pdf': lognorm_pdf[:len(exp_k)]}))  # Debugging output

    # Plot with log-scale x-axis
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stem(exp_k, binom_density, linefmt='b-', markerfmt='bo', basefmt='', label='exp(Binomial) density')
    ax.plot(exp_x_cont, lognorm_pdf, 'r-', linewidth=2, label='LogNormal PDF')
    ax.set_xscale('log')
    ax.set_title(f'exp(Binomial(n={n}, p={p})) vs LogNormal(μ={mean}, σ={std_dev:.2f})')
    ax.set_xlabel('Value (log scale)')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    plt.savefig('logBinomial_logNormal_comparison.png')  # Save the figure for reference
    plt.show()

if __name__ == "__main__":
    n = 50  # number of trials
    p = 0.5  # probability of success
    plot_binom_normal(n, p)
    plot_logBinom_logNormal(n, p)  