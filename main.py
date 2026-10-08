from simulator.data import download_prices
from simulator.estimation import estimate_gbm_params
from simulator.gbm import simulate_gbm
import matplotlib.pyplot as plt

TICKER = "PETR4.SA"

prices = download_prices(TICKER, period = "3y")
mu, sigma = estimate_gbm_params(prices)
s0 = float(prices.iloc[-1])

print(prices.tail())
print(f"Último preço: {s0:.2f}")
print(f"mu = {mu:.2%} | sigma = {sigma:.2%}")

paths = simulate_gbm(s0 = s0, mu = mu, sigma = sigma,
                     horizon = 1, steps = 252, n_sims = 1000, seed = 42)

print("Média dos preços finais: ", paths[-1].mean())
print("Desvio padrão dos preços finais: ", paths[-1].std())

plt.plot(paths[:, :50], alpha = 0.5)
plt.title(f"Monte Carlo: {TICKER}")
plt.xlabel("Dias")
plt.ylabel("Preço")
plt.show()