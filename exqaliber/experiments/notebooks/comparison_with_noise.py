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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from exqaliber.exqaliber_amplitude_estimation import (
    ExqaliberAmplitudeEstimation,
)
from exqaliber.noisy_analytical_sampling import (
    AnalyticalNoisyMLAE,
    run_one_experiment_noisy_iae,
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


results = []
min_noise_magn = -9
max_noise_magn = -1
noise_levels = np.logspace(
    min_noise_magn, max_noise_magn, 10 * (max_noise_magn - min_noise_magn + 1)
)

with concurrent.futures.ProcessPoolExecutor(8) as executor:
    results = list(
        tqdm(
            executor.map(
                run_one_experiment_noisy_iae,
                noise_levels,
                [EXPERIMENT] * len(noise_levels),
            ),
            total=len(noise_levels),
        )
    )

# +
norm = mpl.colors.LogNorm(vmin=noise_levels.min(), vmax=noise_levels.max())
cmap = mpl.cm.get_cmap("viridis", len(noise_levels))

fig, ax = plt.subplots(1)

for res in results:
    intervals = np.array(res.estimate_intervals)
    errors = intervals[:, 1] - intervals[:, 0]

    oracle_queries = 2 * np.array(res.powers) + 1
    x = oracle_queries.cumsum()

    ax.plot(x, errors, c=cmap(norm(res.zeta)), alpha=0.5)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("oracle queries")
ax.set_ylabel("width confidence interval")

fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="noise level"
)

ax.set_title("Iterative amplitude estimation")
# -


#

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
