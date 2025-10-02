import matplotlib.pyplot as plt
import numpy as np

n = 10000
m = np.linspace(1e-6, 1, n)

# tanh
gamma_1 = 1.0
gamma_2 = 1
p_tanh = gamma_1 * np.tanh(gamma_2 * m)

# Max ent
beta = 1
sigma = 10
p_max_ent = np.exp(beta * (m + beta * sigma**2 / 2.0))

# Max ent log
alpha = 0.5
p_max_ent_log = m**alpha

# Max ent log + noise
sigma_m = 1e-6
# sigma_m = 0.0
alpha_eff = alpha * (1 - sigma_m**2 / (2 * m**2))
weights = m ** (alpha_eff)
p = weights / np.sum(weights)


# Plot
plt.close("all")
plt.figure()
plt.plot(m, p_tanh, "-r", label="tanh")
# plt.plot(m, p_max_ent, "bx", label="max ent")
plt.plot(m, p_max_ent_log, "-g", label="max ent log")
plt.plot(m, 2000 * p, "-b", label="max ent log noise")


plt.yscale("log")
plt.legend()
plt.show()
