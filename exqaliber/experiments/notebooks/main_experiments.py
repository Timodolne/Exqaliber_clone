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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] tags=[]
# # Amplitude Estimation experiments

# + tags=[]
"""Statistical amplitude estimation."""
# -

# ## Set-up

# + tags=[]
import os.path

import matplotlib.pyplot as plt
import numpy as np

from exqaliber.analytical_sampling import (
    run_one_experiment_exae,
    run_one_experiment_iae,
    run_one_experiment_mlae,
)
from exqaliber.experiments.amplitude_estimation_experiments import (
    circular_histogram,
    get_results_slice,
    run_experiments_parameters,
)

np.random.seed(1)
# -

# ## Parameters

# ### Noiseless statistical amplitude estimation experiments

# + tags=[]
# saving and running parameters
run_or_load = "load"
save_results = True
show_results = True

# parameters all experiments
epsilon_target = 1e-3
alpha = 1e-2
prior_mean = "true_theta"
prior_std = 1
method = "greedy-smart"
max_iter = 100_000

EXPERIMENT = {
    "epsilon_target": epsilon_target,
    "alpha": alpha,
    "prior_mean": prior_mean,
    "prior_std": prior_std,
    "method": method,
}

# + tags=[]
# parameters sweep
reps = 20
resolution = 4
theta_range = np.linspace(0, np.pi / 2, resolution, endpoint=True)
# replace theta == 0.0 with 2pi
theta_range[0] = 2 * np.pi
epsilon_range = np.logspace(-6, -3, 7)
noise_levels = [0, 1e-6, 1e-3]

parameters = {
    "reps": reps,
    "true_theta": theta_range,
    "zeta": noise_levels,
    "epsilon_target": epsilon_range,
    "max_iter": 100_000,
}
parameters
# -

# # Similar to fig 7 from QCWare paper
#
# ![image.png](attachment:322915d4-5e66-44c2-8483-7b51a51c11c1.png)
#
# _Performance of the QoPrime algorithm under two noise levels, and
# across various choices for the true angle. Shown are both
# theoretical upper bounds (solid) and exact simulated oracle calls
# (dots) for two scenarios: noiseless (blue), and depolarizing rate
# = $10^{−5}$ (green). For each target precision, 20 values of the
# true angle spanning the $[0, \pi/2]$ interval have been selected
# to generate the samples; the horizontal axis represents realized
# approximation precision. The classical Monte Carlo curve (black)
# is obtained by  assuming noiseless classical sampling from a
# constant oracle depth of 1. We see the curve follow a quantum
# $\epsilon^{−1}$ scaling for small errors, which transitions into
# a classical $\epsilon^{−2}$ dependency when the precision is much
# smaller than the noise level ($\epsilon << \gamma$)._
#
#
# Our figure:
#
# $\epsilon$ vs. oracle calls for ExAE, IAE, and MLAE.

# + tags=[]
results_dir = "results/simulations/ExAE-smart/"

results_exae = run_experiments_parameters(
    experiment=EXPERIMENT,
    run_or_load=run_or_load,
    results_dir=results_dir,
    parameters=parameters,
    experiment_f=run_one_experiment_exae,
)

# + tags=[]
results_dir = "results/simulations/IAE/"

results_iae = run_experiments_parameters(
    experiment=EXPERIMENT,
    run_or_load=run_or_load,
    results_dir=results_dir,
    parameters=parameters,
    experiment_f=run_one_experiment_iae,
)

# + tags=[]
# MLAE can't go too deep because of compute limitations
epsilon_target = parameters.get("epsilon_target")
min_epsilon = 5e-5

if not np.all(epsilon_target >= min_epsilon):
    parameters_mlae = parameters.copy()
    print("Changing epsilon target, because MLAE can't go too deep.")
    if isinstance(epsilon_target, np.ndarray):
        epsilon_target = np.concatenate(
            ([min_epsilon], epsilon_target[epsilon_target >= min_epsilon])
        )
    else:
        epsilon_target = max(epsilon_target, min_epsilon)
    parameters_mlae["epsilon_target"] = epsilon_target
else:
    parameters_mlae = parameters.copy()

results_dir = "results/simulations/MLAE/"

results_mlae = run_experiments_parameters(
    experiment=EXPERIMENT,
    run_or_load=run_or_load,
    results_dir=results_dir,
    parameters=parameters,
    experiment_f=run_one_experiment_mlae,
)

# + tags=[]
rules = {"zeta": 0}
plot_kwargs = {"marker": "x"}

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

# EXAE
results_exae_sliced = get_results_slice(results_exae, rules=rules)
abs_epsilons = [
    np.abs(res.final_theta - res.true_theta % (2 * np.pi))
    for k, res in results_exae_sliced.items()
]
oracle_calls = [
    res.num_oracle_queries for k, res in results_exae_sliced.items()
]
ax.scatter(abs_epsilons, oracle_calls, label="ExAE", **plot_kwargs)

# IAE
results_iae_sliced = get_results_slice(results_iae, rules=rules)
abs_epsilons = [
    np.abs(res.estimation_processed - res.true_theta % (2 * np.pi))
    for k, res in results_iae_sliced.items()
]
oracle_calls = [
    res.num_oracle_queries for k, res in results_iae_sliced.items()
]
ax.scatter(abs_epsilons, oracle_calls, label="IAE", **plot_kwargs)

# MLAE
results_mlae_sliced = get_results_slice(results_mlae, rules=rules)
abs_epsilons = [
    np.abs(res.theta - res.true_theta % (2 * np.pi))
    for k, res in results_mlae_sliced.items()
]
oracle_calls = [
    res.num_oracle_queries for k, res in results_mlae_sliced.items()
]
ax.scatter(abs_epsilons, oracle_calls, label="MLAE", **plot_kwargs)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel("Oracle queries")

ax.legend()

# plt.xlim(10**-6, 10**-3)
# -

# # Now with noise
#
# noise levels $\zeta = \{0, 10^{-6}, 10^{-4}\}$


# # Circular histogram of oracle calls

# + tags=[]
parameters = {
    "reps": 300,
    "true_theta": np.linspace(0, np.pi, 180),
    "zeta": 0,
    "epsilon_target": 1e-4,
    "max_iter": 100_000,
}

run_or_load = "load"

# + tags=[]
results_dir = "results/simulations/ExAE-smart/"

results_exae = run_experiments_parameters(
    experiment=EXPERIMENT,
    run_or_load=run_or_load,
    results_dir=results_dir,
    parameters=parameters,
    experiment_f=run_one_experiment_exae,
)

# + tags=[]
if not os.path.exists(f"{results_dir}/figures/"):
    os.mkdir(f"{results_dir}/figures/")

filename = (
    f"{results_dir}/figures/circular_histogram.pdf" if save_results else False
)

circular_histogram(
    results_exae, save=filename, show=show_results, experiment=EXPERIMENT
)
# -
