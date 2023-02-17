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

"""Compare different amplitude estimation algorithms."""
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

# # Amplitude Estimation

# ### Set-up

import matplotlib.pyplot as plt

# +
import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms import (
    AmplitudeEstimation,
    EstimationProblem,
    IterativeAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimation,
)
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector

from exqaliber.exqaliber_amplitude_estimation import (
    ExqaliberAmplitudeEstimation,
)

np.random.seed(0)


# +
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


p = 0.2
A = BernoulliA(p)
Q = BernoulliQ(p)

problem = EstimationProblem(
    state_preparation=A,  # A operator
    grover_operator=Q,  # Q operator
    objective_qubits=[0],
)

exact_p = Statevector(A).probabilities(qargs=[0])[1]

print(f"Exact Probability we're trying to estimate: {exact_p:.5f}")

# +
# quantum sampler
shots = 4

sampler = Sampler(options={"shots": shots})


# -


def pretty_print_result(result):
    """Pretty print the result object."""
    estimation = result.estimation
    num_oracle_queries = result.num_oracle_queries
    confidence_interval = result.confidence_interval
    p_min, p_max = confidence_interval

    message = (
        f"Estimation: {estimation:.4f}. Confidence interval: ",
        f"({p_min:.4f}, {p_max:.4f}).\n",
        f"Oracle queries: {num_oracle_queries}.",
    )

    return "".join(message)


# ## Canonical AE

max_nb_qubits = 16
ae_nb_qubits = np.arange(1, max_nb_qubits + 1)

# # %%time
ae_results = []
for qbts in ae_nb_qubits:
    # the number of evaluation qbts specifies circuit width and accuracy
    ae = AmplitudeEstimation(
        num_eval_qubits=qbts,
        sampler=sampler,
    )
    ae_result = ae.estimate(problem)
    ae_results.append(ae_result)

print("Amplitude Estimation")
print(f"Exact probability: {exact_p:.4f}")
print(pretty_print_result(ae_results[-1]))

# ## Iterative AE

# +
epsilon_target = 1e-4
alpha = 0.01

iae = IterativeAmplitudeEstimation(
    epsilon_target=epsilon_target,  # target accuracy
    alpha=alpha,  # width of the confidence interval
    sampler=sampler,
)
# -

# %%time
iae_result = iae.estimate(problem)

print("Iterative Amplitude Estimation")
print(f"Exact probability: {exact_p:.4f}")
print(pretty_print_result(iae_result))

# ## Maximum Likelihood AE

max_power = 11
mlae_powers = 2 ** np.arange(0, max_power)

# +
# %%time
mlae_results = []

for i, power in enumerate(mlae_powers):
    mlae = MaximumLikelihoodAmplitudeEstimation(
        evaluation_schedule=i, sampler=sampler
    )

    mlae_result = mlae.estimate(problem)

    mlae_results.append(mlae_result)
# -

print("Maximum Likelihood Amplitude Estimation")
print(f"Exact probability: {exact_p:.4f}")
print(pretty_print_result(mlae_results[-1]))

# ## Exqaliber AE

exae = ExqaliberAmplitudeEstimation(
    sampler=sampler,
    epsilon=epsilon_target,
    prior_mean=np.pi / 2,
    prior_std=1,
    alpha=alpha,
)

# %%time
exae_result = exae.estimate(problem)

print("Exqaliber Amplitude Estimation")
print(f"Exact probability: {exact_p:.4f}")
print(pretty_print_result(exae_result))

# # Compare performance

# ## Oracle queries

# +
plt.style.use("ggplot")

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Confidence interval outline
# CAE
conf_intervals = np.array(
    [result.confidence_interval for result in ae_results]
)
ae_powers = 2**ae_nb_qubits
x_ae = np.repeat((2 * ae_powers).cumsum() + 1, shots)
y_1 = np.repeat(conf_intervals[:, 0], shots)
y_2 = np.repeat(conf_intervals[:, 1], shots)

axs[0, 0].fill_between(x_ae, y_1, y_2, alpha=0.5, label="Canonical AE")

# IAE
conf_intervals = np.array(iae_result.estimate_intervals)
iae_oracle_queries = shots * (2 * np.array(iae_result.powers) + 1)
x_iae = iae_oracle_queries.cumsum()
y_1 = conf_intervals[:, 0]
y_2 = conf_intervals[:, 1]

axs[0, 1].fill_between(x_iae, y_1, y_2, alpha=0.5, label="Iterative AE")

# MLAE
conf_intervals = np.array(
    [result.confidence_interval for result in mlae_results]
)
mlae_oracle_queries = 2 * np.array(mlae_powers) + 1
x_mlae = np.repeat(mlae_oracle_queries * shots, shots)
y_1 = np.repeat(conf_intervals[:, 0], shots)
y_2 = np.repeat(conf_intervals[:, 1], shots)

axs[1, 0].fill_between(
    x_mlae, y_1, y_2, alpha=0.5, label="Maximum Likelihood AE"
)

# EXAE
conf_intervals = np.array(exae_result.estimate_intervals)
exae_oracle_queries = 2 * np.array(exae_result.powers) + 1
x_exae = np.repeat(exae_oracle_queries, 1).cumsum()
y_1 = conf_intervals[:, 0]
y_2 = conf_intervals[:, 1]

axs[1, 1].fill_between(x_exae, y_1, y_2, alpha=0.5, label="Exqaliber AE")

# Estimate line
# CAE
y = np.repeat([result.estimation for result in ae_results], shots)
axs[0, 0].plot(x_ae, y)

# IAE
y = np.array(iae_result.estimate_intervals).mean(axis=1)
axs[0, 1].plot(x_iae, y)

# MLAE
y = np.repeat([result.estimation for result in mlae_results], shots)
axs[1, 0].plot(x_mlae, y)

# EXAE
y = exae_result.estimates
axs[1, 1].plot(x_exae, y)

for ax in axs.flatten():
    ax.legend()

    # Axes
    ax.set_xlim(1, 10**4)
    ax.set_xscale("log")
    ax.set_xlabel("Oracle calls")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Estimate of $p$")

# Title
fig.suptitle("Estimation vs. oracle calls")
plt.savefig("results/estimation-oracle-calls.png")

plt.tight_layout()

plt.show()

# +
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

conf_intervals = np.array(exae_result.estimate_intervals)

ax[0].plot(conf_intervals[:, 0], label="min")
ax[0].plot(conf_intervals[:, 1], label="max")
ax[0].plot(exae_result.estimates, label="estimate")

ax[0].set_ylabel("$p$")

ax[0].legend()

theta_intervals = np.array(exae_result.theta_intervals)
thetas = [dist.mean for dist in exae_result.distributions]

ax[1].plot(theta_intervals[:, 0], label="min")
ax[1].plot(theta_intervals[:, 1], label="max")
ax[1].plot(thetas, label="estimate")

ax[1].set_ylabel(r"$\theta$")

ax[1].legend()

plt.suptitle("Discussion of Exqaliber AE conf. interval")

plt.savefig("results/exae-conf-int.png")

plt.show()
# -

# ## Convergence

# +
fig, ax = plt.subplots()

# Convergence
# CAE
conf_intervals = np.array(
    [result.confidence_interval for result in ae_results]
)
ae_powers = 2**ae_nb_qubits
ae_width_conf = conf_intervals[:, 1] - conf_intervals[:, 0]
x_ae = np.repeat((2 * ae_powers).cumsum() + 1, shots)
y_ae = np.repeat(ae_width_conf, shots)
ax.plot(x_ae, y_ae, label="Canonical AE")


# IAE
conf_intervals = np.array(iae_result.estimate_intervals)
iae_oracle_queries = 2 * np.array(iae_result.powers) + 1
iae_width_conf = conf_intervals[:, 1] - conf_intervals[:, 0]
x_iae = np.repeat(iae_oracle_queries, shots).cumsum()
y_iae = np.repeat(iae_width_conf, shots)

ax.plot(x_iae, y_iae, label="Iterative AE")

# MLAE
conf_intervals = np.array(
    [result.confidence_interval for result in mlae_results]
)
mlae_oracle_queries = 2 * np.array(mlae_powers) + 1
mlae_width_conf = conf_intervals[:, 1] - conf_intervals[:, 0]
x_mlae = np.repeat(mlae_oracle_queries * shots, shots)
y_mlae = np.repeat(mlae_width_conf, shots)

ax.plot(x_mlae, y_mlae, label="Maximum Likelihood AE")

# EXAE
conf_intervals = np.array(exae_result.estimate_intervals)
exae_oracle_queries = 2 * np.array(exae_result.powers) + 1
exae_width_conf = conf_intervals[:, 1] - conf_intervals[:, 0]
x_exae = np.repeat(exae_oracle_queries, 1).cumsum()
y_exae = np.repeat(exae_width_conf, 1)

ax.plot(x_exae, y_exae, label="Exqaliber AE")

# Axes
ax.set_xlim(1, None)
ax.set_xscale("log")
ax.set_xlabel("Oracle calls")
ax.set_ylim(epsilon_target / 3, 3)
ax.set_yscale("log")
ax.set_ylabel("Width of confidence interval")

# Lines
x = np.logspace(0, np.log10(ax.get_xlim()[-1]), 100)
y = 10 / x
ax.plot(x, y, label="~1/N", linestyle="--")

y = 1 / np.sqrt(x)
ax.plot(x, y, label=r"~$1/\sqrt{N}$", linestyle="--")

# Title
ax.set_title("Width of confidence interval vs. oracle calls")
ax.legend()

plt.savefig("results/width-confinterval-shots.png")

plt.show()

# +
fig, ax = plt.subplots()

ax.plot(iae_result.powers, label="Iterative AE")
ax.plot(mlae_powers, label="Maximum Likelihood AE")
ax.plot(exae_result.powers, label="Exqaliber AE")

# Axes
ax.set_xlim(0, None)
ax.set_xlabel("Iteration")
ax.set_ylim(0, None)

ax.set_ylabel("$k$")

# Title
ax.set_title("Power of grover operator")
ax.legend()

plt.savefig("results/powers-comparison.png")

plt.show()
