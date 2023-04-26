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
    run_experiment_multiple_thetas,
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
    "epsilon": epsilon_target,
    "alpha": alpha,
    "prior_mean": prior_mean,
    "prior_std": prior_std,
    "method": method,
}

# + tags=[]
# parameters sweep
reps = 1
resolution = 20
max_block_size = 1_000
theta_range = np.linspace(0, np.pi / 2, resolution, endpoint=True)
epsilon_range = np.logspace(-6, -3, 4)
# replace theta == 0.0 with 2pi
theta_range[0] = 2 * np.pi
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
results_dir = f"results/ExAE-smart/{resolution}x{reps}"

EXPERIMENT["output"] = "full"

results_exae = run_experiment_multiple_thetas(
    theta_range,
    experiment=EXPERIMENT,
    run_or_load=run_or_load,
    results_dir=results_dir,
    reps=reps,
    max_iter=max_iter,
    max_block_size=max_block_size,
    experiment_f=run_one_experiment_exae,
    epsilon_range=epsilon_range,
)

# + tags=[]
results_dir = f"results/IAE/{resolution}x{reps}"

results_iae = run_experiment_multiple_thetas(
    theta_range,
    experiment=EXPERIMENT,
    run_or_load=run_or_load,
    results_dir=results_dir,
    reps=reps,
    max_iter=max_iter,
    max_block_size=max_block_size,
    experiment_f=run_one_experiment_iae,
    epsilon_range=epsilon_range,
)

# + tags=[]
results_dir = f"results/MLAE/{resolution}x{reps}"

results_mlae = run_experiment_multiple_thetas(
    theta_range,
    experiment=EXPERIMENT,
    run_or_load=run_or_load,
    results_dir=results_dir,
    reps=reps,
    max_iter=max_iter,
    max_block_size=max_block_size,
    experiment_f=run_one_experiment_mlae,
    epsilon_range=epsilon_range,
)


# + tags=[]
# restoring theta_range[0]
theta_range[0] = 0

# EXAE
abs_epsilons = [
    np.abs(res.final_theta - theta % (2 * np.pi))
    for results_eps in results_exae
    for results_theta, theta in zip(results_eps, theta_range)
    for res in results_theta
]
oracle_calls = [
    res.num_oracle_queries
    for results_exae in results_exae
    for results_theta in results_exae
    for res in results_theta
]
plt.scatter(abs_epsilons, oracle_calls, label="ExAE")

# IAE
abs_epsilons = [
    np.abs(res.estimation_processed - theta % (2 * np.pi))
    for results_eps in results_iae
    for results_theta, theta in zip(results_eps, theta_range)
    for res in results_theta
]
oracle_calls = [
    res.num_oracle_queries
    for results_iae in results_iae
    for results_theta in results_iae
    for res in results_theta
]
plt.scatter(abs_epsilons, oracle_calls, label="IAE")

# MLAE
abs_epsilons = [
    np.abs(res.theta - theta % (2 * np.pi))
    for results_eps in results_mlae
    for results_theta, theta in zip(results_eps, theta_range)
    for res in results_theta
]
oracle_calls = [
    res.num_oracle_queries
    for results_mlae in results_mlae
    for results_theta in results_mlae
    for res in results_theta
]
plt.scatter(abs_epsilons, oracle_calls, label="MLAE")

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$\epsilon$")
plt.ylabel("Oracle queries")

plt.legend()

# plt.xlim(10**-6, 10**-3)
# -

# # Now with noise
#
# noise levels $\zeta = \{0, 10^{-6}, 10^{-4}\}$


# + tags=[]
if not os.path.exists(f"{results_dir}/figures/"):
    os.mkdir(f"{results_dir}/figures/")

filename = (
    f"{results_dir}/figures/circular_histogram.pdf" if save_results else False
)

circular_histogram(
    results_exae,
    theta_range,
    experiment=EXPERIMENT,
    save=filename,
    show=show_results,
)
