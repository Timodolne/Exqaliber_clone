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

import matplotlib as mpl
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
            ([min_epsilon], epsilon_target[epsilon_target > min_epsilon])
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


# + tags=[]
markers = ["x", "o", "v"]
colors = ["b", "r", "g"]

fig, ax = plt.subplots(figsize=(10, 7), dpi=900)

for marker, color, zeta in zip(markers, colors, noise_levels):
    rules = {"zeta": zeta}
    plot_kwargs = {"marker": marker}
    # EXAE
    results_exae_sliced = get_results_slice(results_exae, rules=rules)
    abs_epsilons = [
        np.abs(res.final_theta - res.true_theta % (2 * np.pi))
        for k, res in results_exae_sliced.items()
    ]
    oracle_calls = [
        res.num_oracle_queries for k, res in results_exae_sliced.items()
    ]
    ax.scatter(
        abs_epsilons,
        oracle_calls,
        label="ExAE",
        color=colors[0],
        **plot_kwargs,
    )

    # IAE
    results_iae_sliced = get_results_slice(results_iae, rules=rules)
    abs_epsilons = [
        np.abs(res.estimation_processed - res.true_theta % (2 * np.pi))
        for k, res in results_iae_sliced.items()
    ]
    oracle_calls = [
        res.num_oracle_queries for k, res in results_iae_sliced.items()
    ]
    ax.scatter(
        abs_epsilons, oracle_calls, label="IAE", color=colors[1], **plot_kwargs
    )

    # MLAE
    results_mlae_sliced = get_results_slice(results_mlae, rules=rules)
    abs_epsilons = [
        np.abs(res.theta - res.true_theta % (2 * np.pi))
        for k, res in results_mlae_sliced.items()
    ]
    oracle_calls = [
        res.num_oracle_queries for k, res in results_mlae_sliced.items()
    ]
    ax.scatter(
        abs_epsilons,
        oracle_calls,
        label="MLAE",
        color=colors[2],
        **plot_kwargs,
    )

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel("Oracle queries")

# Create a legend
legend_elements = []
for marker, zeta in zip(markers, noise_levels):
    legend_elements.append(
        ax.scatter([], [], marker=marker, label=zeta, color="black")
    )
for color, ae in zip(colors, ["ExAE", "IAE", "MLAE"]):
    legend_elements.append(
        ax.scatter([], [], marker="s", label=ae, color=color)
    )

ax.legend(handles=legend_elements)


plt.xlim(10**-8, None)
# -

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

# # Converging to the right value

# + tags=[]
noise_levels = np.concatenate(([0], np.logspace(-9, 3, 7)))
theta_range = np.linspace(0, np.pi / 2, 20)

parameters = {
    "reps": 10,
    "true_theta": theta_range,
    "zeta": noise_levels,
    "epsilon_target": 1e-4,
    "max_iter": 100_000,
    "prior_mean": ["true_theta", np.pi / 2],
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
results_dir = "results/simulations/IAE/"

results_iae = run_experiments_parameters(
    experiment=EXPERIMENT,
    run_or_load=run_or_load,
    results_dir=results_dir,
    parameters=parameters,
    experiment_f=run_one_experiment_iae,
)

# + tags=[]
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
rules = {"prior_mean": np.pi / 2}

for theta in theta_range:
    rules["true_theta"] = theta
    results = get_results_slice(results_exae, rules=rules)

    estimates = np.array(
        [res.final_theta for k, res in results.items()]
    ).reshape(-1, results_exae["parameters"]["reps"])
    noise_levels = np.array([res.zeta for k, res in results.items()]).reshape(
        -1, results_exae["parameters"]["reps"]
    )

    estimates_q1 = np.quantile(estimates, 0.25, axis=1)
    estimates_q2 = np.quantile(estimates, 0.5, axis=1)
    estimates_q3 = np.quantile(estimates, 0.75, axis=1)

    err_up = estimates_q2 - estimates_q1
    err_down = estimates_q3 - estimates_q2

    yerr = np.array([err_down, err_up])

    ax.errorbar(
        noise_levels.mean(axis=1),
        estimates_q2,
        yerr=yerr,
        label=theta,
        marker="x",
        capsize=3,
    )

ax.set_xscale("log")
ax.set_title("ExAE prior mean pi/2")

# + tags=[]
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
rules = {"prior_mean": "true_theta"}

for theta in theta_range:
    rules["true_theta"] = theta
    results = get_results_slice(results_exae, rules=rules)

    estimates = np.array(
        [res.final_theta for k, res in results.items()]
    ).reshape(-1, results_exae["parameters"]["reps"])
    noise_levels = np.array([res.zeta for k, res in results.items()]).reshape(
        -1, results_exae["parameters"]["reps"]
    )

    estimates_q1 = np.quantile(estimates, 0.25, axis=1)
    estimates_q2 = np.quantile(estimates, 0.5, axis=1)
    estimates_q3 = np.quantile(estimates, 0.75, axis=1)

    err_up = estimates_q2 - estimates_q1
    err_down = estimates_q3 - estimates_q2

    yerr = np.array([err_down, err_up])

    ax.errorbar(
        noise_levels.mean(axis=1),
        estimates_q2,
        yerr=yerr,
        label=theta,
        marker="x",
        capsize=3,
    )

ax.set_xscale("log")
ax.set_title("ExAE prior mean true theta")

# + tags=[]
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

for theta in theta_range:
    results = get_results_slice(results_iae, rules={"true_theta": theta})

    estimates = np.array(
        [res.estimation_processed for k, res in results.items()]
    ).reshape(-1, results_exae["parameters"]["reps"])
    noise_levels = np.array([res.zeta for k, res in results.items()]).reshape(
        -1, results_exae["parameters"]["reps"]
    )

    estimates_q1 = np.quantile(estimates, 0.25, axis=1)
    estimates_q2 = np.quantile(estimates, 0.5, axis=1)
    estimates_q3 = np.quantile(estimates, 0.75, axis=1)

    err_up = estimates_q2 - estimates_q1
    err_down = estimates_q3 - estimates_q2

    yerr = np.array([err_down, err_up])

    ax.errorbar(
        noise_levels.mean(axis=1),
        estimates_q2,
        yerr=yerr,
        label=theta,
        marker="x",
        capsize=3,
    )

ax.set_xscale("log")
ax.set_title("IAE")
# -

# # Powers of Exqaliber AE

# + tags=[]
noise_levels = np.concatenate(([0], np.logspace(-9, 3, 7)))
theta_range = np.linspace(0, np.pi / 2, 20)

parameters = {
    "reps": 10,
    "true_theta": theta_range,
    "zeta": noise_levels,
    "epsilon_target": 1e-3,
    "max_iter": 100_000,
    "prior_mean": ["true_theta"],
    "output": "powers",
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
norm = mpl.colors.LogNorm(vmin=noise_levels[1], vmax=noise_levels[-1])
cmap = mpl.cm.get_cmap("viridis", len(noise_levels))

fig, ax = plt.subplots(1)

for zeta in noise_levels:
    results = get_results_slice(results_exae, rules={"zeta": zeta})

    abs_epsilons = np.array(
        [
            np.abs(res.final_theta - res.true_theta % (2 * np.pi))
            for k, res in results.items()
        ]
    ).reshape(-1, results_exae["parameters"]["reps"])
    oracle_calls = np.array(
        [res.num_oracle_queries for k, res in results.items()]
    ).reshape(-1, results_exae["parameters"]["reps"])

    ax.scatter(
        abs_epsilons,
        oracle_calls,
        color=cmap(norm(zeta)),
    )

x = np.logspace(-8, 0, 100)
y = 1e5 / x
ax.plot(x, y, label=r"$~ 1/N$", c="r", linestyle="--")

y = 1e5 / (x ** (2))
ax.plot(x, y, label=r"$~ 1/N^2$", c="purple", linestyle="--")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel("Oracle queries")

ax.set_ylim(5e4, 1e9)

ax.legend()

fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="noise level"
)

ax.set_title("Exqaliber amplitude estimation")

# + tags=[]
norm = mpl.colors.LogNorm(vmin=noise_levels[1], vmax=noise_levels[-1])
cmap = mpl.cm.get_cmap("viridis", len(noise_levels))

fig, ax = plt.subplots(1)

for zeta in noise_levels:
    results = get_results_slice(results_exae, rules={"zeta": zeta})

    powers = [res.powers for res in results.values()]

    powers_np = np.empty((len(powers), max([len(arr) for arr in powers])))
    powers_np[:] = np.nan

    for i, arr in enumerate(powers):
        powers_np[i, : len(arr)] = arr

    powers_q1 = np.quantile(powers, 0.25, axis=-1)
    powers_q2 = np.quantile(powers, 0.5, axis=-1)
    powers_q3 = np.quantile(powers, 0.75, axis=-1)

    err_up = powers_q2 - powers_q1
    err_down = powers_q3 - powers_q2

    yerr = np.array([err_down, err_up])

    break

    ax.errorbar(powers_q2, yerr=yerr, c=cmap(norm(zeta)))

# for res in results_one_run:
#     powers = np.array(res.powers)

#     ax.plot(powers, c=cmap(norm(res.zeta)), alpha=0.5)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("iteration")
ax.set_ylabel("power $k$")

fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="noise level"
)

ax.set_title("Exqaliber amplitude estimation")

# + tags=[]
len_thetas = len(results_exae["parameters"]["true_theta"])
reps = results_exae["parameters"]["reps"]
max_iters = max([len(arr) for arr in powers])

powers_np = np.empty((reps, len_thetas, max_iters))
powers_np[:] = np.nan

# for i, arr in enumerate(powers):
#     powers_np[i,:len(arr)] = arr

# + tags=[]
powers_np.shape
# -
