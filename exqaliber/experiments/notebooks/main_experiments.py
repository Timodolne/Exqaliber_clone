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
    format_with_pi,
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
prior_mean = "gaussian"
prior_std = 1
method = "greedy-smart"
max_iter = 1_000_000

EXPERIMENT = {
    "epsilon_target": epsilon_target,
    "alpha": alpha,
    "prior_mean": prior_mean,
    "prior_std": prior_std,
    "method": method,
    "max_iter": max_iter,
}
# -

# # Figure 3
#
# Median number of iterations for different error rates $\varepsilon$.
# Each section of the histogram is a region of width $\pi / 24$ with
# the median number of oracle calls calculated from 30 values of
# true value $\theta_0$ selected uniformly at random. The prior for
# each iteration is taken to be $N(\theta_0, 1)$ and success
# probability $1 - \alpha$ with $\alpha = 0.01$.
#
# * This should use median not mean
# * Use radians
# * Start with 0 on the left
# * Should have lines for other error values
# * Use uniform samples over each $\pi/24$ interval probably 30
# samples per interval. (i.e. 720 samples total)
#
# Optional:
#
# * Change $\theta$ range to $(0, \pi / 2)$ or $(0, 2 \pi)$
#

# + tags=[]
# parameters
reps = 30

# resolution in theta
width = np.pi / 24
bins = np.arange(0, np.pi / 2 + width / 2, width)

# Create an array to hold the samples
true_thetas = np.zeros((len(bins) - 1, reps))

# Draw samples from each distribution
for i in range(len(bins) - 1):
    true_thetas[i] = np.random.uniform(bins[i], bins[i + 1], size=reps)

# true_theta for experiments
true_theta = true_thetas.flatten()

# create parameters dict
parameters = {
    "reps": 1,  # repetition/theta, but that's 1 since we flattened
    "true_theta": true_theta,
    "max_iter": max_iter,
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
def circular_bar(
    results: dict,
    save: bool = False,
    show: bool = True,
    rules: dict = None,
    experiment: dict = None,
    nb_reps: int = None,
    theta_range: np.array = None,
):
    """Plot the circular histogram of nb of queries."""
    try:
        if (
            isinstance(
                results["parameters"]["epsilon_target"],
                (list, tuple, set, np.ndarray),
            )
            and "epsilon_target" not in rules.keys()
        ):
            raise "Choose one epsilon target for circular bar chart."
    except KeyError:
        pass

    if rules is None:
        rules = {"zeta": 0}
    # get queries
    results_sliced = get_results_slice(results, rules=rules)

    thetas = np.array([theta for (theta, i) in results_sliced.keys()])
    queries = np.array(
        [res.num_oracle_queries for res in results_sliced.values()]
    )

    # parameters
    nb_reps = results["parameters"]["reps"] if nb_reps is None else nb_reps
    theta_range = (
        results["parameters"]["true_theta"]
        if theta_range is None
        else theta_range
    )

    thetas = [
        (theta_range[i] + theta_range[i + 1]) / 2
        for i in range(len(theta_range) - 1)
    ]

    # figure
    plt.figure(figsize=(7, 7), dpi=100)
    ax = plt.subplot(projection="polar")

    # data
    width = theta_range[1] - theta_range[0]
    queries = queries.reshape(len(theta_range) - 1, -1)

    # quantiles
    queries_q1 = np.quantile(queries, 0.25, axis=1)
    queries_q2 = np.quantile(queries, 0.5, axis=1)
    queries_q3 = np.quantile(queries, 0.75, axis=1)

    up_err = queries_q3 - queries_q2
    down_err = queries_q2 - queries_q1

    # plot data
    ax.bar(thetas, queries_q2, width=width)
    ax.errorbar(
        thetas,
        queries_q2,
        yerr=[down_err, up_err],
        linestyle="",
        marker="x",
        c="r",
    )

    # axis
    ax.set_xlim(0, np.pi / 2)
    ax.set_rscale("symlog")
    ax.grid(True)

    zero_point = "W"
    match zero_point:
        case "W":
            text_handles = (0.75, 0.95)
        case "E":
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location("E")
            text_handles = (0.05, 0.95)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 12))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 24))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_with_pi))

    # plot experiment
    method = experiment["method"]
    epsilon = experiment["epsilon_target"]
    textstr = f"""
        $\\epsilon={epsilon:.2e}$\n
        $n={nb_reps}$\n
        method={method}
    """

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        *text_handles,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    # Plot title
    title = "Number of iterations before convergence."
    plt.title(title)

    plt.tight_layout(pad=1.0)

    if save:
        plt.savefig(save, dpi=300)

    if show:
        plt.show()


# + tags=[]
circular_bar(
    results_exae,
    save=False,
    show=True,
    rules=None,
    experiment=EXPERIMENT,
    nb_reps=reps,
    theta_range=bins,
)
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

noise_levels = np.concatenate(([0], np.logspace(-9, 3, 7)))

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
