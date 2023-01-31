"""Creates a circular histogram of number of iterations vs theta."""
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from exqaliber.exqaliber_amplitude_estimation import (
    ExqaliberAmplitudeEstimation,
)


def run_experiment_multiple_thetas(theta_range, experiment):
    """Create results for Exqaliber AE for multiple input thetas."""
    # recording
    results = []

    for theta in tqdm(theta_range, desc="theta", position=0, file=sys.stdout):
        results_theta = []
        EXPERIMENT["true_theta"] = theta

        for i in tqdm(
            range(reps),
            desc=" repetitions",
            position=1,
            file=sys.stdout,
            leave=False,
        ):
            # tqdm.write(f'Starting repetition: {i: 2d}.\n', end='')
            ae = ExqaliberAmplitudeEstimation(0.01, 0.01, **EXPERIMENT)

            result = ae.estimate(None)

            results_theta.append(result)

        results.append(results_theta)

    return results


if __name__ == "__main__":

    # parameters
    reps = 50
    resolution = 120
    theta_range = np.linspace(0, 2 * np.pi, resolution)
    prior_mean = np.pi / 4
    prior_std = 0.5
    method = "greedy"
    estimation_problem = None

    run_or_load = "load"

    EXPERIMENT = {
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "method": method,
    }

    if run_or_load == "run":
        results = run_experiment_multiple_thetas(
            theta_range, experiment=EXPERIMENT
        )

        # save results
        with open("results.pkl", "wb") as f:
            pickle.dump(results, f)
        with open("theta_range.pkl", "wb") as f:
            pickle.dump(theta_range, f)

    elif run_or_load == "load":
        # load results
        with open("results.pkl", "rb") as f:
            results = pickle.load(f)
        with open("theta_range.pkl", "rb") as f:
            theta_range = pickle.load(f)

    # get queries
    queries = np.array(
        [[res.num_oracle_queries for res in result] for result in results]
    )
    mean_queries = queries.mean(axis=1)

    # figure
    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = plt.subplot(projection="polar")

    width = 2 * np.pi / resolution
    ax.bar(theta_range, mean_queries, width=width)

    # Plot title
    mu_hat_str = r"$\hat{\mu}$"
    sigma_hat_str = r"$\hat{\sigma}^2$"
    title = (
        "Mean number of iterations before convergence.\n"
        rf"Experiment: {mu_hat_str}: {EXPERIMENT['prior_mean']}, "
        rf"{sigma_hat_str}: {EXPERIMENT['prior_std']}. "
        rf"Method: {EXPERIMENT['method']}"
    )
    plt.title(title)

    plt.tight_layout()
    plt.savefig("circular_histogram.png")

    plt.show()

    print("Done.")
