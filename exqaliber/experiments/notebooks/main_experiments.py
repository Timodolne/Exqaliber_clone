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

# # Amplitude Estimation experiments

# + tags=[]
"""Statistical amplitude estimation."""
# -

# ## Set-up

# + tags=[]
import os.path

import numpy as np

from exqaliber.experiments.amplitude_estimation_experiments import (
    circular_histogram,
    run_experiment_multiple_thetas,
)

# -

# ## Parameters

# ### Noiseless statistical amplitude estimation experiments

# + tags=[]
# saving and running parameters
run_or_load = "run"
save_results = True
show_results = True

# parameters all experiments
epsilon_target = 1e-3
alpha = 1e-2
prior_mean = "uniform"
prior_std = 1
method = "greedy"
max_iter = 100_000

EXPERIMENT = {
    "epsilon": epsilon_target,
    "alpha": alpha,
    "prior_mean": prior_mean,
    "prior_std": prior_std,
    "method": method,
}

# + tags=[]
# parameters theta sweep
reps = 50
resolution = 6
theta_range = np.linspace(0, np.pi, resolution, endpoint=True)
# replace theta == 0.0 with 2pi
theta_range[0] = 2 * np.pi

# + tags=[]
results_dir = f"results/{resolution}x{reps}"

results_multiple_thetas = run_experiment_multiple_thetas(
    theta_range,
    experiment=EXPERIMENT,
    run_or_load=run_or_load,
    results_dir=results_dir,
    reps=reps,
    max_iter=max_iter,
)
# + tags=[]
if not os.path.exists(f"{results_dir}/figures/"):
    os.mkdir(f"{results_dir}/figures/")

filename = (
    f"{results_dir}/figures/circular_histogram.png" if save_results else False
)

circular_histogram(
    results_multiple_thetas,
    theta_range,
    experiment=EXPERIMENT,
    save=filename,
    show=show_results,
)
# -
