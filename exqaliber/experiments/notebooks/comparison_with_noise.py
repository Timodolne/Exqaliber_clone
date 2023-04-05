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

import concurrent.futures

# +
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from exqaliber.experiments.amplitude_estimation_experiments import (
    format_with_pi,
)
from exqaliber.exqaliber_amplitude_estimation import (
    ExqaliberAmplitudeEstimation,
)
from exqaliber.noisy_analytical_sampling import (
    run_one_experiment_noisy_iae,
    run_one_experiment_noisy_mlae,
)

# +
EXPERIMENT = {
    "true_theta": 1.3,
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

min_noise_magn = -6
max_noise_magn = 2
noise_levels = [0] + [
    i
    for i in np.logspace(
        min_noise_magn,
        max_noise_magn,
        5 * (max_noise_magn - min_noise_magn + 1),
    )
]
noise_levels = np.array(noise_levels)

# ## Analytical Iterative AE

# In IAE, the schedule changes.


# +
EXPERIMENT["shots"] = 128
EXPERIMENT["epsilon"] = 1e-5

all_results_iae = []
theta_range = np.linspace(0, np.pi / 2, 13)

for theta in tqdm(theta_range, position=0):
    experiment = copy(EXPERIMENT)
    experiment["true_theta"] = theta
    with concurrent.futures.ProcessPoolExecutor(8) as executor:
        results = list(
            # tqdm(
            executor.map(
                run_one_experiment_noisy_iae,
                noise_levels,
                [experiment] * len(noise_levels),
                # ),
                # total=len(noise_levels),
                # position=1,
                # leave=False
            )
        )
    all_results_iae.append(results)

# +
for i, theta in enumerate(theta_range):
    x = []
    y = []

    for res in all_results_iae[i]:
        x.append(res.zeta)
        y.append(res.estimation_processed)

    plt.plot(x, y, label=format_with_pi(theta))

plt.xlabel("noise level")
plt.ylabel("estimation")

plt.legend()

plt.xscale("log")

plt.title("Estimation by IAE")
# -

# ## Analytical MLAE

# +
experiment = copy(EXPERIMENT)
experiment["m"] = 8

all_results_mlae = []
theta_range = np.linspace(0, np.pi / 2, 13)

for theta in tqdm(theta_range, position=0):
    experiment = copy(EXPERIMENT)
    experiment["true_theta"] = theta
    with concurrent.futures.ProcessPoolExecutor(8) as executor:
        results = list(
            tqdm(
                executor.map(
                    run_one_experiment_noisy_mlae,
                    noise_levels,
                    [experiment] * len(noise_levels),
                ),
                total=len(noise_levels),
                position=1,
                leave=False,
            )
        )
    all_results_mlae.append(results)

# +
for i, theta in enumerate(theta_range):
    x = []
    y = []

    for res in all_results_mlae[i]:
        x.append(res.zeta)
        y.append(res.theta)

    plt.plot(x, y, label=format_with_pi(theta))

plt.xlabel("noise level")
plt.ylabel("estimation")

plt.legend()

plt.xscale("log")

plt.title("Estimation by MLAE")
