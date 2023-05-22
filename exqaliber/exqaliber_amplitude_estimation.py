"""The Exqaliber Quantum Amplitude Estimation Algorithm."""

# from typing import cast

from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numba
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.algorithms.amplitude_estimators import (
    AmplitudeEstimator,
    AmplitudeEstimatorResult,
    EstimationProblem,
)
from qiskit.algorithms.exceptions import AlgorithmError
from qiskit.primitives import BaseSampler
from scipy.optimize import brute
from scipy.stats import norm

from exqaliber.bayesian_updates.distributions.normal import Normal


class ExqaliberAmplitudeEstimation(AmplitudeEstimator):
    r"""The Iterative Amplitude Estimation algorithm.

    This class implements the Exqaliber Quantum Amplitude Estimation
    (EQAE) algorithm, developed by Capgemini Quantum Lab and Cambridge
    Consultants. The output of the algorithm is an estimate that, with
    at least probability :math:`1 - \alpha`, differs by epsilon to the
    target value, where both alpha and epsilon can be specified.

    It is based on Iterative Quantum Amplitude Estimation [1], but
    updates the Grover depth with a Bayes update rul. EQAE iteratively
    applies Grover iterations, selected by maximum variance reduction,
    to find an estimate for the target amplitude.

    References
    ----------
        [1]: Grinko, D., Gacon, J., Zoufal, C., & Woerner, S. (2019).
             Iterative Quantum Amplitude Estimation.
             `arXiv:1912.05559 <https://arxiv.org/abs/1912.05559>`_.
        [2]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
            Quantum Amplitude Amplification and Estimation.
            `arXiv:quant-ph/0005055
            <http://arxiv.org/abs/quant-ph/0005055>`_.
    """

    def __init__(
        self,
        epsilon_target: float = 0.001,
        alpha: float = 0.01,
        sampler: BaseSampler | None = None,
        **kwargs,
    ) -> None:
        r"""
        TODO update docstring.

        epsilon_target: float
            Target precision for estimation target `theta`, has values
            between 0 and 0.5
        alpha: float
            Confidence level, the target probability is 1 - alpha, has
            values between 0 and 1
        sampler: BaseSampler
            A sampler primitive to evaluate the circuits.

        Raises
        ------
            AlgorithmError:
                if the method to compute the confidence
                intervals is not supported
            ValueError: If the target epsilon is not in (0, 0.5]
            ValueError: If alpha is not in (0, 1)
            ValueError: If confint_method is not supported
        """
        # validate ranges of input arguments
        if not 0 < epsilon_target <= 0.5:
            raise ValueError(
                f"The target epsilon must be in (0, 0.5],"
                f"but is {epsilon_target}."
            )

        if not 0 < alpha < 1:
            raise ValueError(
                f"The confidence level alpha must be in (0, 1), but is {alpha}"
            )

        super().__init__()

        # store parameters
        self._epsilon = epsilon_target
        self._alpha = alpha
        self._sampler = sampler
        # self._true_theta = kwargs.get("true_theta", np.pi / 2)
        self._method = kwargs.get("method", "greedy")

        self._prior_mean = kwargs.get("prior_mean", np.pi / 2)
        self._prior_std = kwargs.get("prior_std", 1)

        self._zeta = kwargs.get("zeta", 0)

    @property
    def sampler(self) -> BaseSampler | None:
        """Get the sampler primitive.

        Returns
        -------
            The sampler primitive to evaluate the circuits.
        """
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: BaseSampler) -> None:
        """Set sampler primitive.

        Args
        ----
            sampler: A sampler primitive to evaluate the circuits.
        """
        self._sampler = sampler

    @property
    def epsilon_target(self) -> float:
        """Return the target precision of the algorithm.

        Returns
        -------
            The target precision (which is half the width of the
            confidence interval).
        """
        return self._epsilon

    @epsilon_target.setter
    def epsilon_target(self, epsilon: float) -> None:
        """Set the target precision of the algorithm.

        Args
        ----
            epsilon: Target precision for estimation target `a`.
        """
        self._epsilon = epsilon

    def _find_next_k(
        self, prior_distribution: Normal, method: str = "naive"
    ) -> int:
        """Find the next value of k for the Grover iterator power.

        Args
        ----
            prior_distribution: prior distributions
            method: method for finding next lambda

        Returns
        -------
            The next power k, and boolean flag for the extrapolated
            interval.

        Raises
        ------
            AlgorithmError: if min_ratio is smaller or equal to 1
        """
        analytical_lamda = int(1 / prior_distribution.standard_deviation)
        match method:
            case "naive":
                lamda = analytical_lamda
            case "greedy":
                lamdas = np.arange(1, np.max([2 * analytical_lamda, 200]), 2)
                variance_reduction_factors = Normal.eval_lambdas(
                    lamdas,
                    prior_distribution.mean,
                    prior_distribution.standard_deviation,
                    self._zeta,
                )
                lamda = np.argmax(variance_reduction_factors)
            case "greedy-smart":
                lamdas = np.arange(0, np.max([analytical_lamda, 200]))
                variance_reduction_factors = Normal.eval_lambdas(
                    lamdas,
                    prior_distribution.mean,
                    prior_distribution.standard_deviation,
                    self._zeta,
                )
                lamda = np.argmax(variance_reduction_factors)
            case int():
                n = np.min([analytical_lamda, method])
                lamdas = np.arange(0, 2 * analytical_lamda + n + 1)
                variance_reduction_factors = Normal.eval_lambdas(
                    lamdas,
                    prior_distribution.mean,
                    prior_distribution.standard_deviation,
                    self._zeta,
                )
                lamda = np.argsort(variance_reduction_factors)[::-1][n]
            case _:
                lamda = analytical_lamda

        return np.max([0, int((lamda - 1) / 2)])

    def construct_circuit(
        self, estimation_problem: EstimationProblem, k: int = 0
    ) -> QuantumCircuit:
        r"""Construct the amplitude estimation circuit.

        :math:`\mathcal{Q}^k \mathcal{A} |0\rangle`.

        The A operator is the unitary specifying the QAE problem and
        Q the associated Grover operator.

        estimation_problem:
            The estimation problem for which to construct the QAE
            circuit.
        k:
            The power of the Q operator.

        Returns
        -------
            The circuit implementing
            :math:`\mathcal{Q}^k \mathcal{A} |0\rangle`.
        """
        num_qubits = max(
            estimation_problem.state_preparation.num_qubits,
            estimation_problem.grover_operator.num_qubits,
        )
        circuit = QuantumCircuit(num_qubits, name="circuit")

        # add classical register
        c = ClassicalRegister(len(estimation_problem.objective_qubits))
        circuit.add_register(c)

        # add A operator
        circuit.compose(estimation_problem.state_preparation, inplace=True)

        # add Q^k
        if k != 0:
            circuit.compose(
                estimation_problem.grover_operator.power(k), inplace=True
            )

        # add measurement
        # real hardware can currently not handle operations after
        # measurements, which might happen if the circuit gets
        # transpiled, hence we're adding a safeguard-barrier
        circuit.barrier()
        circuit.measure(estimation_problem.objective_qubits, c[:])

        return circuit

    @staticmethod
    def _compute_mle(
        measurement_results: List[float],
        circuit_depth: List[float],
        error_tol: float = 1e-6,
        plot_results: bool = False,
    ):
        """Compute the MLE for the given schedule and schedule results.

        Converts this form to one suitable for the _compute_fast_mle
        method.

        Parameters
        ----------
        measurement_results : List[float]
            Measurement outcomes for each circuit run. Should be a list
            of {0,1} values.
        circuit_depth : List[float]
            Depth (k) of the corresponding circuits for each
            measurement.
        error_tol: float, optional
            Error tolerance for the final estimate
        plot_results : bool, optional
            Whether to plot the log likelihood function, by default
            False.

        Returns
        -------
        float
            MLE estimate from the measured circuits.
        """
        count_dict = {}

        for i_depth, i_mmt_result in zip(circuit_depth, measurement_results):
            if i_depth not in count_dict:
                count_dict[i_depth] = [0, 0]

            count_dict[i_depth][i_mmt_result] += 1

        return ExqaliberAmplitudeEstimation._compute_fast_mle(
            count_dict, error_tol=error_tol, plot_results=plot_results
        )

    @staticmethod
    def _compute_fast_mle(
        binomial_measurement_results: Dict[int, List[int]],
        error_tol: float = 1e-6,
        plot_results: bool = False,
        true_value: float = None,
    ):
        """Compute the MLE for the given schedule and schedule results.

        Parameters
        ----------
        binomial_measurement_results : Dict[int, List[int]]
            Map of measurement outcomes from a series of binomial
            distributions. Each element is of the form
            depth: [# 0's, # 1's]
        circuit_depth : List[float]
            Depth (k) of the corresponding circuits for each
            measurement.
        error_tol: float, optional
            Error tolerance for the final estimate
        plot_results : bool, optional
            Whether to plot the log likelihood function, by default
            False.
        true_value : float, None
            If plotting the results and this is provided, add a vertical
            line at the true value of theta.

        Returns
        -------
        float
            MLE estimate from the measured circuits.
        """
        # search range
        eps = 1e-15  # to avoid invalid value in log
        search_range = [0 + eps, np.pi / 2 - eps]

        experiment_result_array = np.array(
            [
                [i_depth] + i_results
                for i_depth, i_results in binomial_measurement_results.items()
            ]
        )

        @numba.njit()
        def loglikelihood(theta):
            # loglik contains the first `it` terms of
            # the full loglikelihood
            loglik = np.zeros(1)
            for i_experiment in experiment_result_array:
                angle = (2 * i_experiment[0] + 1) * theta / 2
                loglik = loglik + np.log(np.sin(angle) ** 2) * i_experiment[2]
                loglik = loglik + np.log(np.cos(angle) ** 2) * i_experiment[1]
            return -loglik

        nevals = int(np.pi * 0.5 / error_tol)

        if plot_results:
            est_theta, est_theta_val, x, y = brute(
                loglikelihood, [search_range], Ns=nevals, full_output=True
            )

            plt.plot(x, y)
            plt.axhline(y=est_theta_val, linestyle="--", color="gray")

            if true_value is not None:
                plt.axvline(x=true_value, linestyle="--", color="grey")

            plt.show()
        else:
            est_theta = brute(
                loglikelihood, [search_range], Ns=nevals, full_output=False
            )

        return est_theta[0]

    @staticmethod
    def _compute_mle_variance(
        binomial_measurements: Dict[int, List[int]], mle: float
    ) -> float:
        """Compute the variance of the mle estimator.

        This computes the observed Fisher information at the mle.

        Parameters
        ----------
        binomial_measurement_results : Dict[int, List[int]]
            Map of measurement outcomes from a series of binomial
            distributions. Each element is of the form
            depth: [# 0's, # 1's]
        mle : float
            Maximum likelihood estimator

        Returns
        -------
        float
            Variance of the mle estimator
        """
        fisher_info = 0

        for i_depth, i_mmt in binomial_measurements.items():
            fisher_info += ((2 * i_depth + 1) ** 2 * sum(i_mmt)) / (
                np.sin(mle) ** 4
            )
        return 1 / fisher_info

    def estimate(
        self,
        estimation_problem: Union[EstimationProblem, float],
        output: str = "full",
        max_iter: int = 0,
        post_processing: bool = False,
    ) -> "ExqaliberAmplitudeEstimationResult":
        """Run amplitude estimation algorithm on estimation problem.

        Parameters
        ----------
        estimation_problem : Union[EstimationProblem, float]
            The estimation problem to run. If a float, simulate sampling
            from the probability distribution instead of generating a
            quantum circuit.
        output : Union[str, List[str]] {'sparse', 'full'}, optional
            The level of detail for the returned measurement detail. By
            default 'full', returning all of the information.
            Alternatively, specify a single property or list of
            properties of the Result object.
        max_iter : int, optional
            The maximum number of iterations to run the algorithm for
            before terminating. By default, 0, and runs until the width
            of the interval is to the desired precision.
        post_processing : bool, optional
            Whether to run post-processing on the full set of
            measurement results.

        Returns
        -------
        ExqaliberAmplitudeEstimationResult
            Amplitude estimation results for the algorithm.

        Raises
        ------
            ValueError:
                A quantum instance or Sampler must be provided.
            AlgorithmError:
                Sampler job run error.
        """
        # read estimation problem
        if isinstance(estimation_problem, float):
            self._true_theta = estimation_problem

            if self._prior_mean == "true_theta":
                self._prior_mean = self._true_theta

        if self._prior_mean == "uniform":  # Uniform between 0 and pi
            self._prior_mean = np.pi * np.random.random()
        elif self._prior_mean == "gaussian":  # Gaussian around true_theta
            if hasattr(self, "_true_theta"):
                mu = self._true_theta
            else:
                mu = np.pi / 2
            self._prior_mean = np.random.normal(mu, self._prior_std) % np.pi

        # initiliaze starting variables
        prior = Normal(self._prior_mean, self._prior_std)
        prior_distributions = [prior]
        num_iterations = 0  # keep track of the number of iterations
        sigma_tolerance = self.epsilon_target / norm.ppf(1 - self._alpha / 2)

        # initialize memory variables
        powers = []  # list of powers k: Q^k, (called 'k' in paper)
        measurement_results = []
        binomial_measurements: Dict[int, List[int, int]] = dict()
        num_oracle_queries = 0
        theta_min_0, theta_max_0 = prior.confidence_interval(self._alpha)
        theta_intervals = [[theta_min_0, theta_max_0]]

        # compute a_min_i, a_max_i
        a_min_0 = np.sin(theta_min_0 / 2) ** 2
        a_max_0 = np.sin(theta_max_0 / 2) ** 2
        a_intervals = [[a_min_0, a_max_0]]
        estimates = [np.sin(self._prior_mean / 2) ** 2]

        # do while loop. Theta between 0 and pi.
        while prior_distributions[-1].standard_deviation > sigma_tolerance:
            num_iterations += 1
            if max_iter != 0:
                if num_iterations > max_iter:
                    break

            # get the next k
            k = self._find_next_k(prior_distributions[-1], method=self._method)

            # store the variables
            powers.append(k)

            # set lambda
            lamda = 2 * k + 1

            # Record oracle queries
            num_oracle_queries += lamda

            if self._sampler is not None:
                # run measurements for Q^k A|0> circuit
                circuit = self.construct_circuit(estimation_problem, k)
                try:
                    job = self._sampler.run([circuit], shots=1)
                    ret = job.result()
                except Exception as exc:
                    raise AlgorithmError(
                        "The job was not completed successfully. "
                    ) from exc

                measurement_outcome = max(ret.quasi_dists[0], key=lambda x: x)

                # shots = ret.metadata[0].get("shots")
                # if shots is None:
                #     raise NotImplementedError
                #     circuit = self.construct_circuit(
                #         estimation_problem, k=0, measurement=True
                #     )
                #     try:
                #         job = self._sampler.run([circuit])
                #         ret = job.result()
                #     except Exception as exc:
                #         raise AlgorithmError(
                #             "The job was not completed successfully. "
                #         ) from exc
                #
                #     # calculate the probability of measuring '1'
                #     prob = _probabilities_from_sampler_result(
                #         circuit.num_qubits, ret, estimation_problem
                #     )
                #     # tell MyPy it's a float and not Tuple[int, float]
                #     prob = cast(
                #         float, prob
                #     )
                #     # type: list[float]
                #     a_confidence_interval = [prob, prob]
                #     a_intervals.append(a_confidence_interval)
                #
                #     theta_i_interval = [
                #         np.arccos(1 - 2 * a_i) / 2 / np.pi
                #         for a_i in a_confidence_interval
                #     ]
                #     theta_intervals.append(theta_i_interval)
                #     num_oracle_queries = (
                #         0  # no Q-oracle call, only a single one to A
                #     )
                #     break
                #
                # counts = {
                #     (
                #         np.binary_repr(k, circuit.num_qubits):
                #         round(v * shots)
                #     )
                #     for k, v in ret.quasi_dists[0].items()
                # }
                #
                # # calculate the probability of measuring '1',
                # # 'prob' is a_i in the paper
                # num_qubits = circuit.num_qubits - circuit.num_ancillas
                # # type: ignore
                # one_counts, prob = self._good_state_probability(
                #     estimation_problem, counts, num_qubits
                # )
                #
                # num_one_shots.append(one_counts)
                #
                # # track number of Q-oracle calls
                # num_oracle_queries += shots * k
                #
                # # if on the previous iterations we have K_{i-1}==K_i,
                # # we sum these samples up
                # j = 1  # number of times we stayed fixed at the same K
                # round_shots = shots
                # round_one_counts = one_counts
                # if num_iterations > 1:
                #     while (
                #         (
                #             powers[num_iterations - j] ==
                #             powers[num_iterations]
                #         )
                #         and num_iterations >= j + 1
                #     ):
                #         j = j + 1
                #         round_shots += shots
                #         round_one_counts += num_one_shots[-j]
                # num_oracle_queries += k

            else:  # Cheat sampling
                noise = np.exp(-lamda * self._zeta)
                p = 0.5 * (1 - noise * np.cos(lamda * self._true_theta))
                measurement_outcome = np.random.binomial(1, p)

            measurement_results.append(measurement_outcome)

            if k not in binomial_measurements:
                binomial_measurements[k] = [0, 0]

            binomial_measurements[k][measurement_outcome] += 1

            prior = prior_distributions[-1]

            # Update current belief
            mu, sigma = Normal.update(
                measurement_outcome,
                lamda,
                prior.mean,
                prior.standard_deviation,
                self._zeta,
            )
            posterior = Normal(mu, sigma)

            # Save belief
            prior_distributions.append(posterior)

            # compute theta_min_i, theta_max_i
            theta_min_i, theta_max_i = posterior.confidence_interval(
                self._alpha
            )

            theta_intervals.append([theta_min_i, theta_max_i])

            # compute a_min_i, a_max_i
            a_min_i = np.sin(theta_min_i / 2) ** 2
            a_max_i = np.sin(theta_max_i / 2) ** 2
            a_intervals.append([a_min_i, a_max_i])

            a_i = np.sin(posterior.mean / 2) ** 2
            estimates.append(a_i)

        # get the latest confidence interval for the estimate of theta
        confidence_interval = tuple(a_intervals[-1])

        # the final estimate is coming from the final mean
        final_theta = prior_distributions[-1].mean
        estimation = np.sin(final_theta / 2) ** 2

        result = ExqaliberAmplitudeEstimationResult()
        result.alpha = self._alpha
        result.epsilon_target = self.epsilon_target
        result.zeta = self._zeta
        # result.post_processing = estimation_problem.post_processing

        if post_processing:
            result.mle_estimate = self._compute_fast_mle(
                binomial_measurements,
                error_tol=self.epsilon_target,
                true_value=self._true_theta,
            )
            result.mle_estimate_variance = self._compute_mle_variance(
                binomial_measurements, result.mle_estimate
            )
            result.mle_estimate_epsilon = norm.ppf(
                1 - self._alpha / 2
            ) * np.sqrt(result.mle_estimate_variance)

        result.num_oracle_queries = num_oracle_queries

        result.final_theta = final_theta
        if hasattr(self, "_true_theta"):
            result.true_theta = self._true_theta
        result.estimation = estimation
        result.standard_deviation = prior_distributions[-1].standard_deviation
        result.epsilon_estimated = (
            confidence_interval[1] - confidence_interval[0]
        ) / 2
        result.confidence_interval = confidence_interval

        # if estimation_problem is not None:
        #     result.estimation_processed = (
        #         estimation_problem.post_processing(estimation)
        #     )
        #     confidence_interval = tuple(
        #         estimation_problem.post_processing(x)
        #         for x in confidence_interval
        #     )
        #     result.confidence_interval_processed = confidence_interval
        #     result.epsilon_estimated_processed = (
        #         confidence_interval[1] - confidence_interval[0]
        #     ) / 2

        if output == "full":
            result.estimates = estimates
            result.estimate_intervals = a_intervals
            result.theta_intervals = theta_intervals
            result.distributions = prior_distributions
            result.powers = powers
            result.measurement_results = measurement_results
        elif isinstance(output, str) and output != "sparse":
            setattr(result, output, locals()[output])
        elif isinstance(output, list) and output != "sparse":
            for attr in output:
                setattr(result, output, locals()[attr])

        return result


class ExqaliberAmplitudeEstimationResult(AmplitudeEstimatorResult):
    """The ``ExqaliberAmplitudeEstimation`` result object."""

    def __init__(self) -> None:
        super().__init__()
        self._alpha = None
        self._epsilon_target = None
        self._epsilon_estimated = None
        self._epsilon_estimated_processed = None
        self._estimate_intervals = None
        self._mle_estimate = None
        self._mle_estimate_variance = None
        self._mle_estimate_epsilon = None
        self._theta_intervals = None
        self._powers = None
        self._confidence_interval_processed = None
        self._standard_deviation = None
        self._measurement_results = None

    @property
    def standard_deviation(self) -> float:
        r"""Return the variance of the final estimate."""
        return self._standard_deviation

    @standard_deviation.setter
    def standard_deviation(self, value: float) -> None:
        r"""Set the variance of the final estimate."""
        self._standard_deviation = value

    @property
    def variance(self) -> float:
        r"""Return the variance of the final estimate."""
        return self.standard_deviation**2

    @property
    def alpha(self) -> float:
        r"""Return the confidence level :math:`\alpha`."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        r"""Set the confidence level :math:`\alpha`."""
        self._alpha = value

    @property
    def epsilon_target(self) -> float:
        """Return the target half-width of the confidence interval."""
        return self._epsilon_target

    @epsilon_target.setter
    def epsilon_target(self, value: float) -> None:
        """Set the target half-width of the confidence interval."""
        self._epsilon_target = value

    @property
    def epsilon_estimated(self) -> float:
        """Return the estimated epsilon."""
        return self._epsilon_estimated

    @epsilon_estimated.setter
    def epsilon_estimated(self, value: float) -> None:
        """Set the estimated half-width of the confidence interval."""
        self._epsilon_estimated = value

    @property
    def epsilon_estimated_processed(self) -> float:
        """Return the post-processed epsilon."""
        return self._epsilon_estimated_processed

    @epsilon_estimated_processed.setter
    def epsilon_estimated_processed(self, value: float) -> None:
        """Set the post-processed epsilon."""
        self._epsilon_estimated_processed = value

    @property
    def estimate_intervals(self) -> list[list[float]]:
        """Return conf intervals for the estimate per iteration."""
        return self._estimate_intervals

    @estimate_intervals.setter
    def estimate_intervals(self, value: list[list[float]]) -> None:
        """Set conf intervals for the estimate per iteration."""
        self._estimate_intervals = value

    @property
    def mle_estimate(self) -> float:
        """Return the MLE estimate of the final theta."""
        return self._mle_estimate

    @mle_estimate.setter
    def mle_estimate(self, value: float) -> None:
        """Set the MLE estimate of the final theta."""
        self._mle_estimate = value

    @property
    def mle_estimate_variance(self) -> float:
        """Return the variance of the MLE estimate."""
        return self._mle_estimate_variance

    @mle_estimate_variance.setter
    def mle_estimate_variance(self, val: float) -> None:
        """Set the variance of the MLE estimate."""
        self._mle_estimate_variance = val

    @property
    def mle_estimate_epsilon(self) -> float:
        """Return the epsilon accuracy of the MLE estimate."""
        return self._mle_estimate_epsilon

    @mle_estimate_epsilon.setter
    def mle_estimate_epsilon(self, val: float) -> None:
        """Set the epsilon accuracy of the MLE estimate."""
        self._mle_estimate_epsilon = val

    @property
    def theta_intervals(self) -> list[list[float]]:
        """Return conf intervals for the angles in each iteration."""
        return self._theta_intervals

    @theta_intervals.setter
    def theta_intervals(self, value: list[list[float]]) -> None:
        """Set conf intervals for the angles in each iteration."""
        self._theta_intervals = value

    @property
    def powers(self) -> list[int]:
        """Return powers of the Grover operator in each iteration."""
        return self._powers

    @powers.setter
    def powers(self, value: list[int]) -> None:
        """Set the powers of the Grover operator in each iteration."""
        self._powers = value

    @property
    def confidence_interval_processed(self) -> tuple[float, float]:
        """Return the post-processed confidence interval."""
        return self._confidence_interval_processed

    @confidence_interval_processed.setter
    def confidence_interval_processed(
        self, value: tuple[float, float]
    ) -> None:
        """Set the post-processed confidence interval."""
        self._confidence_interval_processed = value

    @property
    def distributions(self) -> list[Normal]:
        """Return the full list of distributions."""
        return self._distributions

    @distributions.setter
    def distributions(self, distributions: list[Normal]) -> None:
        """Set the list of distributions."""
        self._distributions = distributions

    @property
    def measurement_results(self) -> list[Normal]:
        """Return the full list of measurement results."""
        return self._measurement_results

    @measurement_results.setter
    def measurement_results(self, measurement_results: list[Normal]) -> None:
        """Set the list of measurement results."""
        self._measurement_results = measurement_results


def _chernoff_confint(
    value: float, max_rounds: int, alpha: float
) -> tuple[float, float]:
    """Compute the Chernoff confidence interval for `shots`.

    Uses i.i.d. Bernoulli trials.

    The confidence interval is

        [value - eps, value + eps], where
        eps = sqrt(3 * log(2 * max_rounds/ alpha) / shots)

    but at most [0, 1].


    value:
        The current estimate.
    max_rounds:
        The maximum number of rounds, used to compute epsilon_a.
    alpha:
        The confidence level, used to compute epsilon_a.

    Returns
    -------
        The Chernoff confidence interval.
    """
    eps = np.sqrt(3 * np.log(2 * max_rounds / alpha))
    lower = np.maximum(0, value - eps)
    upper = np.minimum(1, value + eps)
    return lower, upper


if __name__ == "__main__":
    EXPERIMENT = {
        "true_theta": 0.4,
        "prior_mean": np.pi / 2,
        "prior_std": 0.5,
        "method": "greedy",
        "zeta": 1e-9,
    }

    ae = ExqaliberAmplitudeEstimation(0.001, 0.01, **EXPERIMENT)

    result = ae.estimate(EXPERIMENT["true_theta"])

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
