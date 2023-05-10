# ---
# jupyter:
#   jupytext:
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

# + tags=[]
"""Diving in results of ex amplitude estimation."""
# -

# ## Set-up

# + tags=[]

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from exqaliber.analytical_sampling import run_one_experiment_exae
from exqaliber.experiments.amplitude_estimation_experiments import (
    convergence_plot,
    error_in_estimate_2d_hist,
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
reps = 300
resolution = 20
theta_range = np.linspace(0, np.pi, resolution + 1, endpoint=True)
# replace theta == 0.0 with 2pi
theta_range[0] = 2 * np.pi
epsilon_target = np.logspace(-6, -3, 2)
noise_levels = 0

parameters = {
    "reps": reps,
    "true_theta": theta_range,
    "zeta": noise_levels,
    "epsilon_target": epsilon_target,
    "max_iter": 100_000,
}

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
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

for eps in epsilon_target[::-1]:
    rules = {"zeta": 0, "epsilon_target": eps}

    # EXAE
    results_exae_sliced = get_results_slice(results_exae, rules=rules)

    thetas = np.array(
        [res.true_theta for res in results_exae_sliced.values()]
    ).reshape(-1, results_exae["parameters"]["reps"])
    estimations = np.array(
        [res.final_theta for res in results_exae_sliced.values()]
    ).reshape(-1, results_exae["parameters"]["reps"])

    errors = estimations - thetas
    # wrapping around the circle
    thetas = np.mod(thetas, 2 * np.pi)
    errors = np.mod(errors + np.pi, 2 * np.pi) - np.pi
    abs_errors = np.abs(errors)

    epsilons_q1 = np.quantile(abs_errors, 0.25, axis=1)
    epsilons_q2 = np.quantile(abs_errors, 0.5, axis=1)
    epsilons_q3 = np.quantile(abs_errors, 0.75, axis=1)

    err_up = epsilons_q3 - epsilons_q2
    err_down = epsilons_q2 - epsilons_q1

    yerr = np.array([err_down, err_up])

    scatter = ax.errorbar(
        thetas.mean(axis=1),
        epsilons_q2.flatten(),
        yerr=yerr,
        marker="x",
        capsize=3,
        alpha=0.5,
    )
    ax.plot(
        thetas.mean(axis=1),
        [eps] * thetas.shape[0],
        linestyle="--",
        c=scatter[0].get_color(),
        alpha=0.5,
    )

ax.set_yscale("log")
# ax.set_ylim(1e-9, 1e1)

ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$|\epsilon|$")

legend_handles = []
legend_handles.append(
    ax.errorbar(
        [],
        [],
        yerr=[],
        marker="x",
        capsize=3,
        c="black",
        label=r"$\epsilon_{measured}$",
    )
)
legend_handles.append(
    ax.plot([], [], linestyle="--", c="black", label=r"$\epsilon_{target}$")[0]
)

ax.legend(handles=legend_handles)

plt.xlim(0, np.pi + 0.1)
plt.title(
    "Boxplots for errors for different target precisions. Noiseless case. "
    "20 points per theta."
)

# + tags=[]
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

for eps in epsilon_target[::-1]:
    rules = {"zeta": 0, "epsilon_target": eps}

    # EXAE
    results_exae_sliced = get_results_slice(results_exae, rules=rules)

    thetas = np.array(
        [res.true_theta for res in results_exae_sliced.values()]
    ).reshape(-1, results_exae["parameters"]["reps"])
    estimations = np.array(
        [res.final_theta for res in results_exae_sliced.values()]
    ).reshape(-1, results_exae["parameters"]["reps"])

    # errors = estimations - thetas
    # wrapping around the circle
    thetas = np.mod(thetas, 2 * np.pi)
    estimations = np.mod(estimations + 1, 2 * np.pi) - 1
    # abs_errors = np.abs(errors)

    estimations_q1 = np.quantile(estimations, 0.25, axis=1)
    estimations_q2 = np.quantile(estimations, 0.5, axis=1)
    estimations_q3 = np.quantile(estimations, 0.75, axis=1)

    err_up = estimations_q3 - estimations_q2
    err_down = estimations_q2 - estimations_q1

    yerr = np.array([err_down, err_up])

    scatter = ax.errorbar(
        thetas.mean(axis=1),
        estimations_q2.flatten(),
        yerr=yerr,
        # linestyle='',
        marker="x",
        label=rf"$\epsilon_{{target}}={eps:1.0e}$",
        capsize=3,
        alpha=0.5,
    )
    scatter = ax.scatter(
        thetas,
        estimations,
        # linestyle='',
        marker="o",
        s=3,
        alpha=0.5,
    )

ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"Estimate")

ax.legend()

plt.xlim(0, np.pi + 0.1)

# + tags=[]
# parameters sweep
reps = 300
resolution = 20
theta_range = np.linspace(0, np.pi, resolution + 1, endpoint=True)
# replace theta == 0.0 with 2pi
theta_range[0] = 2 * np.pi
epsilon_target = np.logspace(-6, -3, 2)
noise_levels = 0

parameters = {
    "reps": reps,
    "true_theta": theta_range,
    "zeta": noise_levels,
    "epsilon_target": epsilon_target,
    "max_iter": 100_000,
}

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
rules = {"epsilon_target": 1e-6}
val = theta_range[8]

error_in_estimate_2d_hist(
    results_exae,
    save=False,
    show=False,
    experiment=EXPERIMENT,
    rules=rules,
    val=val,
)
plt.show()

# + tags=[]
rules = {"epsilon_target": 1e-3}
val = theta_range[-2]

error_in_estimate_2d_hist(
    results_exae,
    save=False,
    show=False,
    experiment=EXPERIMENT,
    rules=rules,
    val=val,
)
plt.show()
# -

# # Different methods

# + tags=[]
# parameters sweep
run_or_load = "run"
reps = 1
resolution = 10
theta_range = np.linspace(0, np.pi, resolution + 1, endpoint=True)
# replace theta == 0.0 with 2pi
theta_range[0] = 2 * np.pi
epsilon_target = 1e-4  # np.logspace(-6, -3, 2)
noise_levels = 0
method = list(range(0, 10))

parameters = {
    "reps": reps,
    "true_theta": 0.3,
    "zeta": 0,
    "epsilon_target": epsilon_target,
    "max_iter": 100_000,
    "method": method,
    "output": "full",
}

# + tags=[]
results_dir = "results/simulations/ExAE-smart/"

results_exae = run_experiments_parameters(
    experiment=EXPERIMENT,
    run_or_load=run_or_load,
    results_dir=results_dir,
    parameters=parameters,
    experiment_f=run_one_experiment_exae,
)
EXPERIMENT["output"] = "sparse"

# +
rules = {"method": 2}

exae_result = get_results_slice(results_exae, rules=rules)

experiment = EXPERIMENT.copy()
experiment["method"] = rules["method"]
convergence_plot(list(exae_result.values())[0], experiment=experiment)

# + tags=[]
# parameters sweep
reps = 50
resolution = 10
theta_range = np.linspace(0, np.pi, resolution + 1, endpoint=True)
# replace theta == 0.0 with 2pi
theta_range[0] = 2 * np.pi
epsilon_target = 1e-4  # np.logspace(-6, -3, 2)
noise_levels = 0
method = list(range(0, 10))

parameters = {
    "reps": reps,
    "true_theta": theta_range,
    "zeta": 0,
    "epsilon_target": epsilon_target,
    "max_iter": 100_000,
    "method": method,
}

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
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
cmap = mpl.cm.get_cmap("viridis", len(method))

for meth in method[::-1]:
    rules = {"zeta": 0, "method": meth}

    # EXAE
    results_exae_sliced = get_results_slice(results_exae, rules=rules)

    thetas = np.array(
        [res.true_theta for res in results_exae_sliced.values()]
    ).reshape(-1, results_exae["parameters"]["reps"])
    estimations = np.array(
        [res.final_theta for res in results_exae_sliced.values()]
    ).reshape(-1, results_exae["parameters"]["reps"])

    errors = estimations - thetas
    # wrapping around the circle
    thetas = np.mod(thetas, 2 * np.pi)
    errors = np.mod(errors + np.pi, 2 * np.pi) - np.pi
    abs_errors = np.abs(errors)

    epsilons_q1 = np.quantile(abs_errors, 0.25, axis=1)
    epsilons_q2 = np.quantile(abs_errors, 0.5, axis=1)
    epsilons_q3 = np.quantile(abs_errors, 0.75, axis=1)

    err_up = epsilons_q3 - epsilons_q2
    err_down = epsilons_q2 - epsilons_q1

    yerr = np.array([err_down, err_up])

    scatter = ax.errorbar(
        thetas.mean(axis=1),
        epsilons_q2.flatten(),
        yerr=yerr,
        # linestyle='',
        marker="x",
        label=rf"$\epsilon_{{measured}}$ for $n={meth}$",
        capsize=3,
        alpha=0.5,
        c=cmap(meth),
    )

ax.set_yscale("log")
# ax.set_ylim(1e-9, 1e1)

ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$|\epsilon|$")

ax.legend()

plt.xlim(0, np.pi + 0.1)
plt.title(
    "Boxplots for errors for different levels of aggression. Noiseless case. "
    "100 points per theta. Target precision 1e-4"
)

# + tags=[]
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

for meth in method[::-1]:
    rules = {"zeta": 0, "method": meth}

    # EXAE
    results_exae_sliced = get_results_slice(results_exae, rules=rules)

    thetas = np.array(
        [res.true_theta for res in results_exae_sliced.values()]
    ).reshape(-1, results_exae["parameters"]["reps"])
    estimations = np.array(
        [res.final_theta for res in results_exae_sliced.values()]
    ).reshape(-1, results_exae["parameters"]["reps"])

    errors = estimations - thetas
    # wrapping around the circle
    thetas = np.mod(thetas, 2 * np.pi)
    errors = np.mod(errors + np.pi, 2 * np.pi) - np.pi
    abs_errors = np.abs(errors)

    epsilons_q1 = np.quantile(abs_errors, 0.25, axis=1)
    epsilons_q2 = np.quantile(abs_errors, 0.5, axis=1)
    epsilons_q3 = np.quantile(abs_errors, 0.75, axis=1)

    err_up = epsilons_q3 - epsilons_q2
    err_down = epsilons_q2 - epsilons_q1

    yerr = np.array([err_down, err_up])

    scatter = ax.errorbar(
        thetas.mean(axis=1),
        epsilons_q2.flatten(),
        yerr=yerr,
        # linestyle='',
        marker="x",
        label=rf"$\epsilon_{{measured}}$ for $n={meth}$",
        capsize=3,
        alpha=0.5,
        c=cmap(meth),
    )

ax.set_yscale("log")
# ax.set_ylim(1e-9, 1e1)

ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$|\epsilon|$")

ax.legend()

plt.xlim(0, np.pi + 0.1)
plt.ylim(1e-5, 1e-3)
plt.title(
    "Boxplots for errors for different levels of aggression. Noiseless case. "
    "100 points per theta. Target precision 1e-4"
)
# -

# ### Different target precisions

# + tags=[]
# parameters sweep
reps = 10
resolution = 20
theta_range = np.linspace(0, np.pi, resolution + 1, endpoint=True)
# replace theta == 0.0 with 2pi
theta_range[0] = 2 * np.pi
epsilon_target = np.logspace(-6, -3, 4)
noise_levels = 0
method = list(range(0, 10, 3))

parameters = {
    "reps": reps,
    "true_theta": theta_range,
    "zeta": 0,
    "epsilon_target": epsilon_target,
    "max_iter": 100_000,
    "method": method,
}

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
rules = {"zeta": 0}
plot_kwargs = {"marker": "x", "alpha": 0.5}

cmap = mpl.cm.get_cmap("viridis", max(method))

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

for meth in method[::-1]:
    rules = {"zeta": 0, "method": meth}
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
        label=f"n={meth}",
        color=cmap(meth),
        **plot_kwargs,
    )

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel("Oracle queries")

ax.legend()

# plt.xlim(10**-6, 10**-3)

# + tags=[]
# parameters sweep
reps = 10
resolution = 20
theta_range = np.linspace(0, np.pi, resolution + 1, endpoint=True)
# replace theta == 0.0 with 2pi
theta_range[0] = 2 * np.pi
epsilon_target = np.logspace(-6, -3, 7)
noise_levels = 0
method = list(range(0, 10, 3))

parameters = {
    "reps": reps,
    "true_theta": 0.4,
    "zeta": 0,
    "epsilon_target": epsilon_target,
    "max_iter": 100_000,
    "method": method,
}

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
rules = {"zeta": 0}
plot_kwargs = {"marker": "x", "alpha": 0.5}

cmap = mpl.cm.get_cmap("viridis", max(method))

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

for meth in method[::-1]:
    rules = {"zeta": 0, "method": meth}
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
        label=f"n={meth}",
        color=cmap(meth),
        **plot_kwargs,
    )

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel("Oracle queries")

ax.legend()

# plt.xlim(10**-6, 10**-3)
# -
