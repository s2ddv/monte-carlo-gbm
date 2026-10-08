# Monte Carlo Price Simulator

A Python implementation of a Monte Carlo simulator for asset prices, based on **Geometric Brownian Motion (GBM)**. The project estimates model parameters from real historical market data, simulates thousands of possible future price paths, and derives risk metrics such as **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)** from the simulated outcomes.

This repository is part of my self-directed study path in **quantitative finance**. The goal is to build every component from first principles, understanding the mathematics behind each step instead of relying on black-box libraries.

> **Status:** Under active development. See the [Roadmap](#roadmap) for what is implemented and what is planned.

---

## Table of Contents

- [Motivation](#motivation)
- [Features](#features)
- [Mathematical Background](#mathematical-background)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Methodology](#methodology)
- [Roadmap](#roadmap)
- [Limitations](#limitations)
- [Tech Stack](#tech-stack)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Motivation

Monte Carlo methods are a cornerstone of quantitative finance. They are used for derivatives pricing, portfolio risk management, scenario analysis, and stress testing. When closed-form solutions do not exist, simulation provides a flexible and general way to approximate the distribution of future outcomes.

This project aims to:

1. Implement a GBM-based price simulator that is correct, vectorized, and reproducible.
2. Calibrate the model with real market data (expected return and volatility).
3. Quantify uncertainty using statistical tools: confidence intervals, percentiles, and tail-risk measures.
4. Serve as a solid foundation for more advanced work: option pricing, correlated multi-asset simulation, and variance reduction techniques.

---

## Features

**Core (in scope for the first release)**

- Simulation of asset price paths under Geometric Brownian Motion
- Vectorized implementation with NumPy for performance
- Reproducible results through seeded random number generation
- Parameter estimation (annualized drift and volatility) from historical prices
- Historical data download via `yfinance`
- Visualization of simulated paths and the terminal price distribution
- Summary statistics: expected price, standard deviation, percentiles, and confidence intervals
- Risk metrics: Value at Risk (VaR) and Conditional Value at Risk (CVaR / Expected Shortfall)

**Planned**

- European option pricing via Monte Carlo, validated against the Black-Scholes closed-form solution
- Variance reduction techniques (antithetic variates, control variates)
- Convergence analysis (error versus number of simulations)
- Multi-asset simulation with correlated returns (Cholesky decomposition)
- Command-line interface and configuration files

---

## Mathematical Background

### Geometric Brownian Motion

Under GBM, the asset price $S_t$ follows the stochastic differential equation:

$$
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t
$$

where:

| Symbol | Meaning |
|--------|---------|
| $S_t$ | Asset price at time $t$ |
| $\mu$ | Expected annualized return (drift) |
| $\sigma$ | Annualized volatility |
| $W_t$ | Standard Wiener process (Brownian motion) |

Applying Itô's lemma yields the exact solution:

$$
S_t = S_0 \exp\left( \left(\mu - \frac{\sigma^2}{2}\right) t + \sigma W_t \right)
$$

### Discretization

For a time step $\Delta t$, prices are simulated recursively with the exact discretization:

$$
S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{\sigma^2}{2}\right) \Delta t + \sigma \sqrt{\Delta t} \, Z \right), \qquad Z \sim \mathcal{N}(0, 1)
$$

Because this is the exact solution of the SDE, it introduces no discretization error, unlike the Euler-Maruyama scheme.

### Parameter Estimation

Given a series of historical prices, daily log returns are computed as:

$$
r_t = \ln\left(\frac{S_t}{S_{t-1}}\right)
$$

With 252 trading days per year, the annualized parameters are estimated as:

$$
\hat{\sigma} = \sqrt{252} \cdot \text{std}(r_t), \qquad \hat{\mu} = 252 \cdot \text{mean}(r_t) + \frac{\hat{\sigma}^2}{2}
$$

### Monte Carlo Estimation and Error

For $N$ independent simulations, the expected value of a quantity $X$ is estimated by the sample mean $\bar{X}$. The standard error shrinks at a rate of $1/\sqrt{N}$:

$$
\text{SE} = \frac{s}{\sqrt{N}}
$$

where $s$ is the sample standard deviation. This is why convergence analysis and variance reduction techniques matter in practice.

### Risk Metrics

- **Value at Risk (VaR)** at confidence level $\alpha$: the loss threshold that is not exceeded with probability $\alpha$ over the chosen horizon, estimated as the corresponding percentile of the simulated loss distribution.
- **Conditional Value at Risk (CVaR)**: the expected loss given that the loss exceeds the VaR threshold. It captures tail risk beyond VaR.

---

## Project Structure

```
monte-carlo-price-simulator/
├── src/
│   └── simulator/
│       ├── __init__.py
│       ├── data.py          # Historical data download and cleaning
│       ├── estimation.py    # Drift and volatility estimation
│       ├── gbm.py           # GBM path simulation
│       ├── risk.py          # VaR, CVaR and summary statistics
│       └── plotting.py      # Visualization utilities
├── notebooks/               # Exploratory analysis and experiments
├── tests/                   # Unit tests
├── main.py                  # Example entry point
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

> The structure above reflects the intended organization and may evolve as the project grows.

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- `pip` and `venv`

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/monte-carlo-price-simulator.git
cd monte-carlo-price-simulator

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

The snippet below illustrates the core simulation logic: 1,000 one-year paths with 252 daily steps.

```python
import numpy as np
import matplotlib.pyplot as plt

S0, mu, sigma = 100.0, 0.10, 0.20     # initial price, annual drift, annual volatility
T, steps, n_sims = 1.0, 252, 1000
dt = T / steps

rng = np.random.default_rng(seed=42)
z = rng.standard_normal((steps, n_sims))

log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
paths = S0 * np.exp(np.cumsum(log_returns, axis=0))

plt.plot(paths[:, :50], alpha=0.5)
plt.title("Monte Carlo Simulation: 50 GBM Price Paths")
plt.xlabel("Trading Days")
plt.ylabel("Price")
plt.show()
```

Calibrating the model with real market data:

```python
import numpy as np
import yfinance as yf

prices = yf.download("PETR4.SA", period="3y")["Close"].squeeze()
log_returns = np.log(prices / prices.shift(1)).dropna()

sigma = log_returns.std() * np.sqrt(252)
mu = log_returns.mean() * 252 + 0.5 * sigma**2
```

---

## Methodology

1. **Data collection:** download adjusted historical prices for the chosen asset.
2. **Calibration:** compute log returns and estimate annualized drift and volatility.
3. **Simulation:** generate `N` price paths over the chosen horizon using the exact GBM discretization.
4. **Analysis:** compute the distribution of terminal prices, confidence intervals, VaR, and CVaR.
5. **Validation:** compare Monte Carlo estimates against analytical results (for example, the known mean of a log-normal distribution) and check convergence as `N` increases.

---

## Roadmap

- [x] Project setup and repository structure
- [ ] GBM path simulator (vectorized)
- [ ] Historical data download and parameter estimation
- [ ] Visualization of paths and terminal distribution
- [ ] VaR and CVaR computation
- [ ] Unit tests and validation against analytical results
- [ ] Convergence analysis
- [ ] European option pricing and comparison with Black-Scholes
- [ ] Variance reduction (antithetic and control variates)
- [ ] Correlated multi-asset simulation

---

## Limitations

GBM is a simple and widely used model, but it relies on assumptions that real markets do not fully satisfy:

- **Constant volatility:** real volatility is stochastic and clusters over time.
- **Normally distributed log returns:** empirical returns exhibit fat tails and skewness.
- **No jumps:** sudden price gaps (earnings, macro events) are not captured.
- **Constant drift:** expected returns are estimated from the past and are inherently noisy.

Extensions such as stochastic volatility models (Heston), jump-diffusion models (Merton), and GARCH-based simulation are natural next steps beyond the scope of this project.

---

## Tech Stack

- [Python](https://www.python.org/)
- [NumPy](https://numpy.org/): vectorized numerical computing
- [pandas](https://pandas.pydata.org/): data handling
- [SciPy](https://scipy.org/): statistics and distributions
- [Matplotlib](https://matplotlib.org/): visualization
- [yfinance](https://github.com/ranaroussi/yfinance): historical market data

---

## Disclaimer

This project is for **educational and research purposes only**. It does not constitute financial advice, and its outputs should not be used to make investment decisions. Simulated results depend on model assumptions and historical data, and do not predict actual future prices.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
