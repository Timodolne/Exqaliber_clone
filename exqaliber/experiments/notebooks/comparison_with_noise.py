# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: exqaliber
#     language: python
#     name: python3
# ---

"""Compare different noisy amplitude estimation algorithms."""

# # Noisy amplitude estimation

# ## Exqaliber AE

import time
from copy import copy

# +
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from exqaliber.exqaliber_amplitude_estimation import (
    ExqaliberAmplitudeEstimation,
)
from exqaliber.noisy_analytical_sampling import (
    AnalyticalNoisyIAE,
    AnalyticalNoisyMLAE,
)

# +
EXPERIMENT = {
    "true_theta": 0.4,
    "prior_mean": "true_theta",
    "prior_std": 0.5,
    "method": "greedy",
    "zeta": 1e-5,
}

ae = ExqaliberAmplitudeEstimation(0.01, 0.01, **EXPERIMENT)
estimation_problem = None

result = ae.estimate(estimation_problem)

print(f"Executed {len(result.powers)} rounds")
print(
    f"Finished with standard deviation of {result.standard_deviation:.6f} "
    f"and mean {result.estimation:.6f}, "
    f"(true theta: {EXPERIMENT['true_theta']})."
)
print(
    f"Finished with epsilon {result.epsilon_estimated:.6f} and estimate "
    f"{result.estimation:.6f}. Target epsilon was {result.epsilon_target}."
)

print("Done.")


# -

# ## Analytical Iterative AE

# In IAE, the schedule changes.


# +

results = []
min_noise_magn = -9
max_noise_magn = -1
noise_levels = np.logspace(
    min_noise_magn, max_noise_magn, max_noise_magn - min_noise_magn + 1
)

min_noise = 1e-3
max_noise = 1e-2
noise_levels = np.linspace(min_noise, max_noise, 10)

experiment = copy(EXPERIMENT)

for noise_level in tqdm(noise_levels):
    np.random.seed(0)
    print(noise_level)
    experiment["zeta"] = noise_level

    iae = AnalyticalNoisyIAE(0.05, 0.01, **experiment)

    time.sleep(0.1)

    results.append(iae.estimate(experiment["true_theta"]))
# -

for i, noise_level in enumerate(noise_levels):
    plt.plot(results[i].powers, label=noise_level)
plt.legend()

for i, noise_level in enumerate(noise_levels):
    plt.plot(results[i].estimate_intervals, label=noise_level)
plt.legend()

# ## MLAE

# +
experiment = copy(EXPERIMENT)

noise_level = 0.2
experiment["zeta"] = noise_level

noisy_mlae = AnalyticalNoisyMLAE(10, **experiment)

res = noisy_mlae.estimate(0.8, alpha=0.01)
theta = 2 * np.arcsin(np.sqrt(res.estimation))
# -

res.estimation_processed

res.confidence_interval
