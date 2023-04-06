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

# +
import concurrent.futures
from copy import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from exqaliber.experiments.amplitude_estimation_experiments import (
    format_with_pi,
)
from exqaliber.noisy_analytical_sampling import (
    run_one_experiment_noisy_exae,
    run_one_experiment_noisy_iae,
    run_one_experiment_noisy_mlae,
)

# -

# # Noisy amplitude estimation

# +
EXPERIMENT = {"shots": 128, "epsilon": 1e-5, "alpha": 0.01}

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
# -

# ## Analytical Iterative AE

# In IAE, the schedule changes.


# +
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

experiment = copy(EXPERIMENT)
experiment["m"] = 8

# +

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
# -

# ## Exqaliber AE

EXPERIMENT["prior_mean"] = "true_theta"
EXPERIMENT["prior_std"] = 0.5
EXPERIMENT["method"] = "greedy"

# +
experiment = {
    "true_theta": 0.4,
    "prior_mean": "true_theta",
    "prior_std": 0.5,
    "method": "greedy",
    "zeta": 1e-5,
}


with concurrent.futures.ProcessPoolExecutor(8) as executor:
    results_one_run = list(
        tqdm(
            executor.map(
                run_one_experiment_noisy_exae,
                noise_levels,
                [experiment] * len(noise_levels),
            ),
            total=len(noise_levels),
            position=1,
            leave=False,
        )
    )

# +
norm = mpl.colors.LogNorm(vmin=noise_levels[1], vmax=noise_levels[-1])
cmap = mpl.cm.get_cmap("viridis", len(noise_levels))

fig, ax = plt.subplots(1)

for res in results_one_run:
    intervals = np.array(res.estimate_intervals)
    errors = intervals[:, 1] - intervals[:, 0]

    oracle_queries = 2 * np.array(res.powers) + 1
    x = oracle_queries.cumsum()

    ax.plot(x, errors, c=cmap(norm(res.zeta)), alpha=0.5)

x = np.logspace(1, 6, 100)
y = 10 / x
ax.plot(x, y, label=r"$~ 1/N$", c="r", linestyle="--")

y = 1 / (x ** (0.5))
ax.plot(x, y, label=r"$~ 1/\sqrt{N}$", c="purple", linestyle="--")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("oracle queries")
ax.set_ylabel("width confidence interval")

ax.legend()

fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="noise level"
)

ax.set_title("Exqaliber amplitude estimation")


# +
norm = mpl.colors.LogNorm(vmin=noise_levels[1], vmax=noise_levels[-1])
cmap = mpl.cm.get_cmap("viridis", len(noise_levels))

fig, ax = plt.subplots(1)

for res in results_one_run:
    powers = np.array(res.powers)

    ax.plot(powers, c=cmap(norm(res.zeta)), alpha=0.5)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("iteration")
ax.set_ylabel("power $k$")

fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="noise level"
)

ax.set_title("Exqaliber amplitude estimation")


# +
all_results_exae = []
theta_range = np.linspace(0, np.pi / 2, 13)

for theta in tqdm(theta_range, position=0):
    experiment = copy(EXPERIMENT)
    experiment["true_theta"] = theta
    with concurrent.futures.ProcessPoolExecutor(8) as executor:
        results = list(
            tqdm(
                executor.map(
                    run_one_experiment_noisy_exae,
                    noise_levels,
                    [experiment] * len(noise_levels),
                ),
                total=len(noise_levels),
                position=1,
                leave=False,
            )
        )
    all_results_exae.append(results)

# +
for i, theta in enumerate(theta_range):
    x = []
    y = []

    for res in all_results_exae[i]:
        x.append(res.zeta)
        y.append(res.theta)

    plt.plot(x, y, label=format_with_pi(theta))

plt.xlabel("noise level")
plt.ylabel("estimation")

plt.legend()

plt.xscale("log")

plt.title("Estimation by EXAE")
