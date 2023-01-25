"""The Exqaliber Quantum Amplitude Estimation Algorithm."""

from typing import cast

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.algorithms.amplitude_estimators import (
    AmplitudeEstimator,
    AmplitudeEstimatorResult,
    EstimationProblem,
)
from qiskit.algorithms.amplitude_estimators.ae_utils import (
    _probabilities_from_sampler_result,
)
from qiskit.algorithms.exceptions import AlgorithmError
from qiskit.primitives import BaseSampler

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
        epsilon_target: float,
        alpha: float,
        sampler: BaseSampler | None = None,
        **kwargs,
    ) -> None:
        r"""
        TODO update docstring.

        epsilon_target: float
            Target precision for estimation target `a`, has values
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
        self._true_theta = kwargs.get("true_theta")
        self._method = kwargs.get("method")

        self._prior_mean = kwargs.get("prior_mean", 0.5)
        self._prior_variance = kwargs.get("prior_variance", 0.5)

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
                lamdas = np.arange(0, np.max([2 * analytical_lamda, 200]))
                variance_reduction_factors = Normal.eval_lambdas(
                    lamdas,
                    prior_distribution.mean,
                    prior_distribution.standard_deviation,
                )
                lamda = np.argmax(variance_reduction_factors)
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

    def estimate(
        self, estimation_problem: EstimationProblem
    ) -> "ExqaliberAmplitudeEstimationResult":
        """Run amplitude estimation algorithm on estimation problem.

        Args
        ----
            estimation_problem: The estimation problem.

        Returns
        -------
            An amplitude estimation results object.

        Raises
        ------
            ValueError: A quantum instance or Sampler must be provided.
            AlgorithmError: Sampler job run error.
        """
        # initialize memory variables
        powers = [0]  # list of powers k: Q^k, (called 'k' in paper)
        # a priori knowledge of theta / 2 / pi
        theta_intervals = [[0, 1.0]]
        a_intervals = [
            [0.0, 1.0]
        ]  # a priori knowledge of the confidence interval of the estimate
        num_oracle_queries = 0
        num_one_shots = []
        prior_distributions = [Normal(self._prior_mean, self._prior_variance)]

        # initiliaze starting variables
        # TODO find some way of making this a variable
        var_tolerance = 0.000001

        num_iterations = 0  # keep track of the number of iterations

        # do while loop, keep in mind that we scaled theta mod 2pi
        # such that it lies in [0,1]
        while prior_distributions[-1].variance > var_tolerance:
            num_iterations += 1

            # get the next k
            k = self._find_next_k(prior_distributions[-1], method=self._method)

            # store the variables
            powers.append(k)

            if estimation_problem is not None:
                # run measurements for Q^k A|0> circuit
                circuit = self.construct_circuit(estimation_problem, k)
                try:
                    job = self._sampler.run([circuit], shots=1)
                    ret = job.result()
                except Exception as exc:
                    raise AlgorithmError(
                        "The job was not completed successfully. "
                    ) from exc

                shots = ret.metadata[0].get("shots")
                if shots is None:
                    raise NotImplementedError
                    circuit = self.construct_circuit(
                        estimation_problem, k=0, measurement=True
                    )
                    try:
                        job = self._sampler.run([circuit])
                        ret = job.result()
                    except Exception as exc:
                        raise AlgorithmError(
                            "The job was not completed successfully. "
                        ) from exc

                    # calculate the probability of measuring '1'
                    prob = _probabilities_from_sampler_result(
                        circuit.num_qubits, ret, estimation_problem
                    )
                    prob = cast(
                        float, prob
                    )  # tell MyPy it's a float and not Tuple[int, float ]

                    a_confidence_interval = [prob, prob]  # type: list[float]
                    a_intervals.append(a_confidence_interval)

                    theta_i_interval = [
                        np.arccos(1 - 2 * a_i) / 2 / np.pi
                        for a_i in a_confidence_interval
                    ]
                    theta_intervals.append(theta_i_interval)
                    num_oracle_queries = (
                        0  # no Q-oracle call, only a single one to A
                    )
                    break

                counts = {
                    np.binary_repr(k, circuit.num_qubits): round(v * shots)
                    for k, v in ret.quasi_dists[0].items()
                }

                # calculate the probability of measuring '1',
                # 'prob' is a_i in the paper
                num_qubits = circuit.num_qubits - circuit.num_ancillas
                # type: ignore
                one_counts, prob = self._good_state_probability(
                    estimation_problem, counts, num_qubits
                )

                num_one_shots.append(one_counts)

                # track number of Q-oracle calls
                num_oracle_queries += shots * k

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

            else:  # Cheat sampling
                lamda = 4 * k + 2
                p = 0.5 * (1 - np.cos(lamda * self._true_theta))
                measurement_outcome = np.random.binomial(1, p)
                num_oracle_queries += k

            prior = prior_distributions[-1]

            # Update current belief
            mu, var = Normal.update(
                measurement_outcome,
                lamda,
                prior.mean,
                prior.standard_deviation,
            )
            posterior = Normal(mu, var)
            # TODO decide what should go in .update, std or var

            # Save belief
            prior_distributions.append(posterior)

            # TODO confidence interval calculations
            # # compute a_min_i, a_max_i
            # a_i_min, a_i_max = _chernoff_confint(prob, round_shots,
            # max_rounds, self._alpha)

            # # compute theta_min_i, theta_max_i
            # if upper_half_circle:
            #     theta_min_i = np.arccos(1 - 2 * a_i_min) / 2 / np.pi
            #     theta_max_i = np.arccos(1 - 2 * a_i_max) / 2 / np.pi
            # else:
            #     theta_min_i = 1-np.arccos(1 - 2 * a_i_max) / 2 / np.pi
            #     theta_max_i = 1-np.arccos(1 - 2 * a_i_min) / 2 / np.pi

            # # compute theta_u, theta_l of this iteration
            # scaling = 4 * k + 2  # current K_i factor
            # theta_u = (
            #   int(scaling * theta_intervals[-1][1]) + theta_max_i)
            #    / scaling
            # theta_l = (
            #   int(scaling * theta_intervals[-1][0]) + theta_min_i)
            # / scaling
            # theta_intervals.append([theta_l, theta_u])
            #
            # # compute a_u_i, a_l_i
            # a_u = np.sin(2 * np.pi * theta_u) ** 2
            # a_l = np.sin(2 * np.pi * theta_l) ** 2
            # a_u = cast(float, a_u)
            # a_l = cast(float, a_l)
            # a_intervals.append([a_l, a_u])

        # # get the latest confidence interval for the estimate of a
        # confidence_interval = tuple(a_intervals[-1])

        # the final estimate is the mean of the confidence interval
        estimation = prior_distributions[-1].mean

        result = ExqaliberAmplitudeEstimationResult()
        result.alpha = self._alpha
        # result.post_processing = estimation_problem.post_processing
        result.num_oracle_queries = num_oracle_queries

        result.estimation = estimation
        result.variance = prior_distributions[-1].variance
        result.distributions = prior_distributions
        # result.epsilon_estimated =
        # (confidence_interval[1] - confidence_interval[0]) / 2
        # result.confidence_interval = confidence_interval

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
        #     result.estimate_intervals = a_intervals
        #     result.theta_intervals = theta_intervals
        result.powers = powers

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
        self._theta_intervals = None
        self._powers = None
        self._confidence_interval_processed = None
        self._variance = None

    @property
    def variance(self) -> float:
        r"""Return the variance of the final estimate."""
        return self._variance

    @variance.setter
    def variance(self, value: float) -> None:
        r"""Set the variance of the final estimate."""
        self._variance = value

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
        "prior_mean": 0.38,
        "prior_variance": 0.03,
        "method": "greedy",
    }

    ae = ExqaliberAmplitudeEstimation(0.01, 0.01, **EXPERIMENT)
    estimation_problem = None

    result = ae.estimate(estimation_problem)

    print(f"Executed {len(result.powers)} rounds")
    print(f"Finished with variance of {result.variance}")
