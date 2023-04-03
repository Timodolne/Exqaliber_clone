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

from typing import cast

import matplotlib.pyplot as plt

# +
import numpy as np
from qiskit.algorithms import (
    IterativeAmplitudeEstimation,
    IterativeAmplitudeEstimationResult,
)
from qiskit.algorithms.amplitude_estimators.iae import (
    _chernoff_confint,
    _clopper_pearson_confint,
)

from exqaliber.exqaliber_amplitude_estimation import (
    ExqaliberAmplitudeEstimation,
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


class AnalyticalIAE(IterativeAmplitudeEstimation):
    """Analytically sampling with Iterative AE."""

    def __init__(self, epsilon_target, alpha, *args, **kwargs):
        super().__init__(epsilon_target, alpha)
        self._true_theta = kwargs.get("true_theta", np.pi / 2)
        self._zeta = kwargs.get("zeta", 0)

    def estimate(self, true_theta):
        """Estimate with analytical sampling."""
        # initialize memory variables

        # list of powers k: Q^k, (called 'k' in paper)
        powers = [0]
        # list of multiplication factors (called 'q' in paper)
        ratios = []
        # a priori knowledge of theta / 2 / pi
        theta_intervals = [[0, 1 / 4]]
        # a priori knowledge of the confidence interval of the estimate
        a_intervals = [[0.0, 1.0]]
        num_oracle_queries = 0
        num_one_shots = []

        # maximum number of rounds
        max_rounds = (
            int(
                np.log(self._min_ratio * np.pi / 8 / self._epsilon)
                / np.log(self._min_ratio)
            )
            + 1
        )
        # initially theta is in the upper half-circle
        upper_half_circle = True

        num_iterations = 0  # keep track of the number of iterations
        # number of shots per iteration
        shots = 0
        # do while loop, keep in mind that we scaled
        # theta mod 2pi such that it lies in [0,1]
        while (
            theta_intervals[-1][1] - theta_intervals[-1][0]
            > self._epsilon / np.pi
        ):
            num_iterations += 1

            # get the next k
            k, upper_half_circle = self._find_next_k(
                powers[-1],
                upper_half_circle,
                theta_intervals[-1],  # type: ignore
                min_ratio=self._min_ratio,
            )

            # store the variables
            powers.append(k)
            ratios.append((2 * powers[-1] + 1) / (2 * powers[-2] + 1))

            # set lambda
            lamda = 2 * k + 1

            noise = np.exp(-lamda * self._zeta)
            p = 0.5 * (1 - noise * np.cos(lamda * self._true_theta))
            measurement_outcome = np.random.binomial(1, p)

            one_counts = measurement_outcome
            prob = measurement_outcome

            num_one_shots.append(one_counts)

            # track number of Q-oracle calls
            num_oracle_queries += shots * k

            # if on the previous iterations we have K_{i-1} == K_i,
            # we sum these samples up
            j = 1  # number of times we stayed fixed at the same K
            round_shots = 1
            round_one_counts = one_counts
            if num_iterations > 1:
                while (
                    powers[num_iterations - j] == powers[num_iterations]
                    and num_iterations >= j + 1
                ):
                    j = j + 1
                    round_shots += 1
                    round_one_counts += num_one_shots[-j]

            # compute a_min_i, a_max_i
            if self._confint_method == "chernoff":
                a_i_min, a_i_max = _chernoff_confint(
                    prob, round_shots, max_rounds, self._alpha
                )
            else:  # 'beta'
                a_i_min, a_i_max = _clopper_pearson_confint(
                    round_one_counts, round_shots, self._alpha / max_rounds
                )

            # compute theta_min_i, theta_max_i
            if upper_half_circle:
                theta_min_i = np.arccos(1 - 2 * a_i_min) / 2 / np.pi
                theta_max_i = np.arccos(1 - 2 * a_i_max) / 2 / np.pi
            else:
                theta_min_i = 1 - np.arccos(1 - 2 * a_i_max) / 2 / np.pi
                theta_max_i = 1 - np.arccos(1 - 2 * a_i_min) / 2 / np.pi

            # compute theta_u, theta_l of this iteration
            scaling = 4 * k + 2  # current K_i factor
            theta_u = (
                int(scaling * theta_intervals[-1][1]) + theta_max_i
            ) / scaling
            theta_l = (
                int(scaling * theta_intervals[-1][0]) + theta_min_i
            ) / scaling
            theta_intervals.append([theta_l, theta_u])

            # compute a_u_i, a_l_i
            a_u = np.sin(2 * np.pi * theta_u) ** 2
            a_l = np.sin(2 * np.pi * theta_l) ** 2
            a_u = cast(float, a_u)
            a_l = cast(float, a_l)
            a_intervals.append([a_l, a_u])

        # get the latest confidence interval for the estimate of a
        confidence_interval = tuple(a_intervals[-1])

        # the final estimate is the mean of the confidence interval
        estimation = np.mean(confidence_interval)

        result = IterativeAmplitudeEstimationResult()
        result.alpha = self._alpha
        # result.post_processing = estimation_problem.post_processing
        result.num_oracle_queries = num_oracle_queries

        result.estimation = estimation
        result.epsilon_estimated = (
            confidence_interval[1] - confidence_interval[0]
        ) / 2
        result.confidence_interval = confidence_interval

        # result.estimation_processed =
        # estimation_problem.post_processing(estimation)
        # confidence_interval = tuple(
        #     estimation_problem.post_processing(x)
        #     for x in confidence_interval
        # )
        # result.confidence_interval_processed = confidence_interval
        result.epsilon_estimated_processed = (
            confidence_interval[1] - confidence_interval[0]
        ) / 2
        result.estimate_intervals = a_intervals
        result.theta_intervals = theta_intervals
        result.powers = powers
        result.ratios = ratios

        return result


np.random.seed(0)
iae = AnalyticalIAE(0.01, 0.01, **EXPERIMENT)
res = iae.estimate(0.1)

print(res.estimation)

# +

plt.plot(res.powers)
# -

plt.plot(np.array(res.estimate_intervals))
