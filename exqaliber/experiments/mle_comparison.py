"""Generate figure for impact of using MLE post-processing with SAE."""
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from exqaliber.analytical_sampling import run_one_experiment_exae
from exqaliber.exqaliber_amplitude_estimation import (
    ExqaliberAmplitudeEstimationResult,
)


def get_experiment_data(**kwargs) -> List[ExqaliberAmplitudeEstimationResult]:
    """Get experiment data for a choice of theta and precision.

    Kwargs
    ------
    alpha : float
        Final confidence interval of SAE wihout MLE will be a (1-alpha)
        width, by default 0.01.
    prior_mean_method : str, {'true_theta', 'uniform', 'gaussian'}
        Method to pick the prior mean. Recommended to use 'true_theta'
        which uses the true theta value, by default 'true_theta'.
    prior_std : float
        Standard deviation of the prior, by default 1
    lambda_selection_method : str, {'naive', 'greedy', 'greedy-smart'}
        Method to pick the next value of lambda, by default
        'greedy-smart' which picks the maximiser of the variance
        reduction factor that lies to the left of approximately 1/std.
    max_sae_iterations : int
        The maximum number of iterations to run a single experiment for,
        by default 1e+6.
    post_processing : bool
        Whether to calculate the mle value of each experiment, by
        default True
    bin_width : int
        Width of the bins to pick theta values from
    samples_per_bin : int
        The number of samples to pick per bin
    epsilon_target : List[float]
        Target precision values

    Returns
    -------
    List[ExqaliberAmplitudeEstimationResult]
        Results corresponding to the experiments run
    """
    np.random.seed(1)

    # parameters all experiments
    alpha = kwargs.get("alpha", 1e-2)
    prior_mean = kwargs.get("prior_mean_method", "true_theta")
    prior_std = kwargs.get("priod_std", 1)
    method = kwargs.get("lambda_selection_method", "greedy-smart")
    max_iter = kwargs.get("max_sae_iterations", 1e6)

    EXPERIMENT = {
        "alpha": alpha,
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "method": method,
        "max_iter": max_iter,
        "post_processing": kwargs.get("post_processing", True),
    }

    # resolution in theta
    bin_width = kwargs.get("bin_width", np.pi / 24)
    samples_per_bin = kwargs.get("samples_per_bin", 5)
    bins = np.arange(2 * bin_width, np.pi / 2 + bin_width / 2, bin_width)

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
        "epsilon_target": kwargs.get("epsilon_target", [1e-3, 1e-4, 1e-5]),
    }

    results: List[ExqaliberAmplitudeEstimationResult] = []

    for i_eps_target in parameters["epsilon_target"]:
        EXPERIMENT["epsilon_target"] = i_eps_target
        for j_theta_batch in true_thetas:
            for k_theta in j_theta_batch:
                EXPERIMENT["true_theta"] = k_theta
                results.append(run_one_experiment_exae(EXPERIMENT))

        print(f"{i_eps_target} completed")

    return results


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


def plot_figure_exae_actual_prec_vs_expected_prec() -> None:
    """Plot 'fig::exae-actual-prec-vs-expected-prec'.

    Actual precision of the final estimate for SAE with and without MLE
    post-processing. We sample theta_0 Theta_0 from a uniform
    distribution of x for target precisions of epsilon = 10^-3, 10^-4,
    ..., 10^-7. The prior for each iteration is taken to be
    N(theta_0, 1) and success probability 1 - alpha with alpha = 0.01.

    This is a scatter plot where we would expect to see 99% of the
    values below the diagonal line if this was a true credible region.
    """
    results = get_experiment_data(
        samples_per_bin=5, epsilon_target=[1e-2, 1e-3, 1e-4, 1e-5]
    )

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

    plot_sae_vs_mle_comparision(exae_prec, exae_post_proc_prec)


if __name__ == "__main__":
    plot_figure_exae_actual_prec_vs_expected_prec()
