"""Generate figure for impact of using MLE post-processing with SAE."""
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from exqaliber.analytical_sampling import run_one_experiment_exae
from exqaliber.exqaliber_amplitude_estimation import (
    ExqaliberAmplitudeEstimationResult,
)


def plot_sae_vs_mle_comparision(
    sae_results: np.ndarray, sae_with_mle_results: np.ndarray
) -> None:
    """Plot the actual precision vs expected precision for SAE with MLE.

    Parameters
    ----------
    sae_results : np.ndarray
        Pairs of results for the original Statistical Amplitude
        Estimation routine with the first column the expected precision,
        and the second column the corresponding actual precision i.e.
        half the width of the confidence interval vs. the absolute error
    sae_with_mle_results : np.ndarray
        Pairs of results for the Statistical Amplitude Estimation
        routine with mle, the first column is the expected precision,
        and the second column the corresponding actual precision i.e.
        half the width of the confidence interval vs. the absolute error
    """
    plt.scatter(
        sae_results[:, 0], sae_results[:, 1], color="blue", label="Exqalibur"
    )
    plt.scatter(
        sae_with_mle_results[:, 0],
        sae_with_mle_results[:, 1],
        color="red",
        label="Exqalibur w/ MLE",
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.plot([0, 1], [0, 1], "--", color="grey")

    plt.xlabel(r"Expected precision: $\varepsilon$")
    plt.ylabel(r"Actual precision: $|\theta_0 - \hat{\theta}|$")

    plt.legend()

    print(
        "Original algorithm produces a more accurate estimate on average "
        f"{np.mean(sae_results[:,1] < sae_with_mle_results[:,1])}"
    )
    print(
        "Actual theta is within the confidence interval of original algorithm "
        f"on average {np.mean(sae_results[:,0] > sae_results[:,1])}"
    )
    print(
        "Actual theta is within the confidence interval of mle algorithm on "
        f"average "
        f"{np.mean(sae_with_mle_results[:,0] > sae_with_mle_results[:,1])}"
    )

    plt.show()


np.random.seed(1)

# parameters all experiments
alpha = 1e-2
prior_mean = "true_theta"
prior_std = 1
method = "greedy-smart"
max_iter = 1_000_000

EXPERIMENT = {
    "alpha": alpha,
    "prior_mean": prior_mean,
    "prior_std": prior_std,
    "method": method,
    "max_iter": max_iter,
    "post_processing": True,
}

# resolution in theta
bin_width = np.pi / 24
bins = np.arange(2 * bin_width, np.pi / 2 + bin_width / 2, bin_width)
samples_per_bin = 1

# Create an array to hold the samples
true_thetas = np.zeros((len(bins) - 1, samples_per_bin))

# Draw samples from each distribution
for i in range(len(bins) - 1):
    true_thetas[i] = np.random.uniform(
        bins[i], bins[i + 1], size=samples_per_bin
    )

# true_theta for experiments
true_theta = true_thetas.flatten()

# create parameters dict
parameters = {
    "reps": 1,  # repetition/theta, but that's 1 since we flattened
    "true_theta": true_theta,
    "max_iter": max_iter,
    "epsilon_target": [1e-2, 1e-3, 1e-4, 1e-5],
}

results: List[ExqaliberAmplitudeEstimationResult] = []


for i_eps_target in parameters["epsilon_target"]:
    EXPERIMENT["epsilon_target"] = i_eps_target
    for j_theta_batch in true_thetas:
        for k_theta in j_theta_batch:
            EXPERIMENT["true_theta"] = k_theta
            results.append(run_one_experiment_exae(EXPERIMENT))

    print(f"{i_eps_target} completed")


exae_prec = np.array(
    [
        (
            i_result.epsilon_estimated,
            abs(i_result.true_theta - i_result.final_theta),
        )
        for i_result in results
    ]
)
exae_post_proc_prec = np.array(
    [
        (
            i_result.epsilon_estimated,
            abs(i_result.true_theta - i_result.mle_estimate),
        )
        for i_result in results
    ]
)
