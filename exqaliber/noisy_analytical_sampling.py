"""Noisy Analytical Sampling based on qiskit implementations."""

from typing import cast

import numpy as np
from qiskit.algorithms import (
    IterativeAmplitudeEstimation,
    IterativeAmplitudeEstimationResult,
    MaximumLikelihoodAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimationResult,
)
from qiskit.algorithms.amplitude_estimators.iae import (
    _chernoff_confint,
    _clopper_pearson_confint,
)
from qiskit.algorithms.amplitude_estimators.mlae import (
    _compute_fisher_information,
    _fisher_confint,
)


def post_processing(x):
    """Post processing p and theta."""
    return np.arcsin(np.sqrt(np.min([1, x])))


class AnalyticalNoisyIAE(IterativeAmplitudeEstimation):
    """Analytically (noisy) sampling with Iterative AE."""

    def __init__(self, epsilon_target, alpha, *args, **kwargs):
        super().__init__(epsilon_target, alpha)
        self._shots = kwargs.get("shots", 1024)
        self._zeta = kwargs.get("zeta", 0)

    def estimate(self, true_theta, max_iterations=10000):
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
        shots = self._shots
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
            lamda = 4 * k + 2

            noise = np.exp(-lamda * self._zeta)
            p = 0.5 * (1 - noise * np.cos(lamda * true_theta))
            measurement_outcome = np.random.binomial(1, p, shots)

            one_counts = measurement_outcome.sum()
            prob = np.mean(measurement_outcome)

            num_one_shots.append(one_counts)

            # track number of Q-oracle calls
            num_oracle_queries += shots * (2 * k + 1)

            # if on the previous iterations we have K_{i-1} == K_i,
            # we sum these samples up
            j = 1  # number of times we stayed fixed at the same K
            round_shots = shots
            round_one_counts = one_counts
            if num_iterations > 1:
                while (
                    powers[num_iterations - j] == powers[num_iterations]
                    and num_iterations >= j + 1
                ):
                    j = j + 1
                    round_shots += shots
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

            if num_iterations > max_iterations:
                break

        # get the latest confidence interval for the estimate of a
        confidence_interval = tuple(a_intervals[-1])

        # the final estimate is the mean of the confidence interval
        estimation = np.mean(confidence_interval)

        result = IterativeAmplitudeEstimationResult()
        result.alpha = self._alpha
        result.zeta = self._zeta
        result.true_theta = true_theta

        result.post_processing = post_processing
        result.num_oracle_queries = num_oracle_queries

        result.estimation = estimation
        result.epsilon_estimated = (
            confidence_interval[1] - confidence_interval[0]
        ) / 2
        result.confidence_interval = confidence_interval

        result.estimation_processed = post_processing(estimation)
        confidence_interval = tuple(
            post_processing(x) for x in confidence_interval
        )
        result.confidence_interval_processed = confidence_interval
        result.epsilon_estimated_processed = (
            confidence_interval[1] - confidence_interval[0]
        ) / 2
        result.estimate_intervals = a_intervals
        estimate_intervals_processed = []
        for bounds in a_intervals:
            estimate_intervals_processed.append(
                [post_processing(b) for b in bounds]
            )
        result.estimate_intervals_processed = estimate_intervals_processed
        result.theta_intervals = theta_intervals
        theta_intervals_processed = []
        for bounds in theta_intervals:
            theta_intervals_processed.append(
                [post_processing(b) for b in bounds]
            )
        result.theta_intervals_processed = theta_intervals_processed
        result.powers = powers
        result.ratios = ratios

        return result


def run_one_experiment_noisy_iae(noise, experiment):
    """Run one noisy iae experiment."""
    np.random.seed(0)
    experiment["zeta"] = noise
    epsilon = experiment.get("epsilon", 0.01)
    alpha = experiment.get("alpha", 0.01)
    noisy_iae = AnalyticalNoisyIAE(epsilon, alpha, **experiment)

    true_theta = experiment["true_theta"]

    return noisy_iae.estimate(true_theta)


class AnalyticalNoisyMLAE(MaximumLikelihoodAmplitudeEstimation):
    """Analytically (noisy) sampling with Iterative AE."""

    def __init__(self, evaluation_schedule, shots=128, *args, **kwargs):
        super().__init__(evaluation_schedule)
        self._zeta = kwargs.get("zeta", 0)
        self._shots = shots

    def compute_mle(self, circuit_results):
        """Compute maximum likelihood estimate."""
        good_counts = np.mean(np.array(circuit_results), axis=1)
        all_counts = np.ones_like(good_counts)

        # search range
        eps = 1e-15  # to avoid invalid value in log
        search_range = [0 + eps, np.pi / 2 - eps]

        def loglikelihood(theta):
            # loglik contains the first `it` terms of
            # the full loglikelihood
            loglik = 0
            for i, k in enumerate(self._evaluation_schedule):
                angle = (2 * k + 1) * theta
                loglik += np.log(np.sin(angle) ** 2) * good_counts[i]
                loglik += np.log(np.cos(angle) ** 2) * (
                    all_counts[i] - good_counts[i]
                )
            return -loglik

        est_theta = self._minimizer(loglikelihood, [search_range])

        return est_theta, good_counts

    def estimate(self, true_theta, alpha):
        """Estimate theta with MLAE."""
        result = MaximumLikelihoodAmplitudeEstimationResult()
        result.evaluation_schedule = self._evaluation_schedule
        result.zeta = self._zeta
        # result.minimizer = self._minimizer
        result.post_processing = post_processing

        circuit_results = []
        for k in self._evaluation_schedule:
            lamda = 4 * k + 2
            noise = np.exp(-lamda * self._zeta)
            p = 0.5 * (1 - noise * np.cos(lamda * true_theta))
            measurement_outcome = np.random.binomial(1, p, self._shots)

            circuit_results.append(measurement_outcome)

        result.circuit_results = circuit_results
        result.shots = self._shots

        # run maximum likelihood estimation
        theta, good_counts = self.compute_mle(result.circuit_results)

        # store results
        result.theta = theta
        result.good_counts = good_counts
        result.estimation = np.sin(result.theta) ** 2

        # not sure why pylint complains, this is a callable
        # and the tests pass
        # pylint: disable=not-callable
        result.estimation_processed = result.post_processing(result.estimation)

        result.fisher_information = _compute_fisher_information(result)
        result.num_oracle_queries = result.shots * sum(
            k for k in result.evaluation_schedule
        )

        # compute and store confidence interval
        confidence_interval = _fisher_confint(result, alpha, observed=False)
        result.confidence_interval = confidence_interval
        # result.confidence_interval_processed = tuple(
        #     result.post_processing(value) for value in
        # confidence_interval
        # )

        return result


def run_one_experiment_noisy_mlae(noise, experiment):
    """Run one noisy mlae experiment."""
    np.random.seed(0)
    experiment["zeta"] = noise
    evaluation_schedule = experiment.get("m", 6)
    noisy_mlae = AnalyticalNoisyMLAE(evaluation_schedule, **experiment)

    true_theta = experiment["true_theta"]
    alpha = experiment.get("alpha", 0.01)

    return noisy_mlae.estimate(true_theta, alpha=alpha)
