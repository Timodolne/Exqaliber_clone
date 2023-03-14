"""Exqaliber amplitude estimation on a quantum computer."""

import numpy as np
from qiskit.algorithms import EstimationProblem
from qiskit.circuit import QuantumCircuit
from qiskit_aer.primitives import Sampler

from exqaliber.experiments.amplitude_estimation_experiments import (
    convergence_plot,
)
from exqaliber.exqaliber_amplitude_estimation import (
    ExqaliberAmplitudeEstimation,
)


class BernoulliA(QuantumCircuit):
    """A circuit representing the Bernoulli A operator."""

    def __init__(self, probability):
        super().__init__(1)  # circuit on 1 qubit

        theta_p = 2 * np.arcsin(np.sqrt(probability))
        self.ry(theta_p, 0)


class BernoulliQ(QuantumCircuit):
    """A circuit representing the Bernoulli Q operator."""

    def __init__(self, probability):
        super().__init__(1)  # circuit on 1 qubit

        self._theta_p = 2 * np.arcsin(np.sqrt(probability))
        self.ry(2 * self._theta_p, 0)

    def power(self, k):
        """Implement the efficient power of Q."""
        q_k = QuantumCircuit(1)
        q_k.ry(2 * k * self._theta_p, 0)
        return q_k


if __name__ == "__main__":
    # params
    p = 0.2
    A = BernoulliA(p)
    Q = BernoulliQ(p)

    EXPERIMENT = {
        "true_theta": 2 * np.arcsin(np.sqrt(p)),
        "prior_mean": np.pi / 4,
        "prior_std": 0.5,
        "method": "greedy",
    }

    problem = EstimationProblem(
        state_preparation=A,  # A operator
        grover_operator=Q,  # Q operator
        objective_qubits=[0],
        # "good" state Psi1 is identified as measuring |1> in qubit 0
    )

    # quantum instance
    sampler = Sampler()

    ae = ExqaliberAmplitudeEstimation(
        0.01, 0.01, sampler=sampler, **EXPERIMENT
    )

    result = ae.estimate(problem)

    print(f"Executed {len(result.powers)} rounds")
    print(
        f"Finished with standard deviation of {result.standard_deviation:.6f} "
        f"and mean {result.estimation:.6f}, "
        f"(true theta: {EXPERIMENT['true_theta']})."
    )

    convergence_plot(
        result,
        experiment=EXPERIMENT,
        save=False,
        show=True,
    )

    print("Done.")
