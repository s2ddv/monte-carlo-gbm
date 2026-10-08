from simulator.gbm import simulate_gbm
import matplotlib.pyplot as plt

paths = simulate_gbm(s0 = 100, mu = 0.10, sigma = 0.20,
                     horizon = 1, steps = 252, n_sims = 1000, seed = 42)

print(paths.shape)
print(paths[0, :5])
print(paths.min() > 0)
print(paths[-1].mean())
print(paths[-1].std())

plt.plot(paths[:, :50], alpha = 0.5)
plt.xlabel("Dias")
plt.ylabel("Preço")
plt.show()