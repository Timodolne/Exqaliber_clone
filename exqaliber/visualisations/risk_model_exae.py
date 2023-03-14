# ---
# jupyter:
#   jupytext:
#     formats: py,ipynb
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

"""Exqaliber risk modelling on a quantum computer."""

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # Exqaliber risk modelling on a quantum computer
#
# Risk is everywhere, both in your everyday life (e.g. running a yellow
# light), and in business (e.g. the political situation in a country).
# Risks can be hard to model, due to their probabilistic nature.
# Stochastics events, luckily, form an interesting opportunity for
# a new upcoming technology: quantum computing. Quantum computers are
# probabilistic by nature and are an excellent candidate for risk
# modelling.
#
# In this notebook, we follow some of the efforts of \[1\] to model
# business risk on a quantum computer. We use various amplitude
# estimation techniques (from IBM's qiskit) to model business risks,
# ultimately testing a new amplitude estimation algorithm: Exqaliber.
# To build the risk model, we follow the logic from
# [this blogpost](https://tinyurl.com/59r7htwc).
#
#
# ### References
#
# \[1\] Braun, M. C., Decker, T., Hegemann, N., Kerstan, S. F., &
# Schäfer, C. (2021). _A Quantum Algorithm for the Sensitivity Analysis
# of Business Risks_. http://arxiv.org/abs/2103.05475
# -

# # Create risk model and state preperation
#
# Following [this blogpost](https://tinyurl.com/59r7htwc) and Braun
# et al \[1\], the risk model in the figure below is used to model,
# in this case, business risk.
#
# ![climaterisk-small.png](attachment:f8a741c7-158c-4c59-9b09-0baa6048a398.png)
#
# For this demo, we're interested in the probability that the
# total impact exceeds a value of 12.

# ## Set-up

# +
import matplotlib.pyplot as plt
import numpy as np
from qiskit import Aer, QuantumCircuit, QuantumRegister
from qiskit.algorithms import (
    AmplitudeEstimation,
    EstimationProblem,
    IterativeAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimation,
)
from qiskit.circuit.library import GroverOperator, WeightedAdder
from qiskit.quantum_info import Statevector
from qiskit_aer.primitives import Sampler
from tqdm.notebook import tqdm

from exqaliber.exqaliber_amplitude_estimation import (
    ExqaliberAmplitudeEstimation,
)

np.random.seed(0)
# -

# ## Setting up probabilities

# +
# The intrinsic probabilities of the risk items
intrinsic_p = [0.8, 0.2, 0.1, 0.05]

# The transition probabilities
transition_p = [[0, 0, 0, 0], [0, 0, 0.5, 0.4], [0, 0, 0, 0], [0, 0, 0, 0]]

# The impacts
impact = [0, 1, 4, 8]

# Create a list of the probbilities combined
# (i,j), p_ij, pos or neg
probabilities = [
    ((0, 0), 0.8, 0),
    ((0, 1), 1, 0),
    ((1, 2), intrinsic_p[2], 0),
    (
        (1, 2),
        transition_p[1][2] + (1 - transition_p[1][2]) * intrinsic_p[2],
        1,
    ),
    ((1, 3), intrinsic_p[3], 0),
    (
        (1, 3),
        transition_p[1][3] + (1 - transition_p[1][3]) * intrinsic_p[3],
        1,
    ),
]


# -

# ## Creating the quantum state
#
# The quantum state will first implement the risk model as a quantum
# state, then, using Qiskit's ```WeightedAdder```, the total impact is
# calculated. Finally, the qubits representing the number '12' in the
# sum are used to mark a qubit, which now is in the state
#
# $$ |\psi_{\text{marking qubit}}\rangle = P(\text{total impact} < 12)
# |0\rangle + P(\text{total impact} \geq 12)|1\rangle. $$


def state_preparation_func(probabilities, impact, threshold=12, adder=None):
    """
    Prepare the state from a probabilites list.

    This function implements the risk model from a list of probabilities
    and adds up all the impacts.
    To make the risk model, the logic from the aforementioned blogpost
    is used.
    """
    # Helper converter function
    def p_to_theta(x):
        return 2 * np.arcsin(np.sqrt(x))

    if type(impact) != list:
        impact = np.array(impact).tolist()

    assert len(intrinsic_p) == len(
        impact
    ), "Arrays should have the same length"

    # Get nb_risk_items
    nb_risk_items = len(intrinsic_p)

    # Account for different ordering of qubits in qiskit
    impact = impact[::-1]

    if isinstance(adder, type(None)):
        # Create the weighted adder, summing the values of the qubits
        adder = WeightedAdder(num_state_qubits=len(impact), weights=impact)

    # Create a marking qubit
    marking_qubit = QuantumRegister(1, "mark")

    # Start the quantum circuit
    state_preparation = QuantumCircuit(
        *adder.qregs, marking_qubit, name="state_preparation"
    )

    # This part is the risk model
    for probability in probabilities:
        (i, j), p_ij, sign = probability

        # Account for different ordering of qubits in qiskit
        i = (nb_risk_items - 1) - i
        j = (nb_risk_items - 1) - j

        if i == j:  # It's an intrinsic probability
            state_preparation.rx(p_to_theta(p_ij), i)
            continue
        if p_ij == 1:  # Part of an XOR
            state_preparation.cx(i, j, ctrl_state=sign)
            continue
        state_preparation.crx(p_to_theta(p_ij), i, j, ctrl_state=sign)

    # This part adds up the financial impact
    state_preparation.append(adder.to_gate(), adder.qubits)

    # This code finds out which qubits in the sum-register encode
    # the threshold
    threshold_qbts = (
        np.argwhere(np.array(list(bin(threshold)[2:][::-1]), dtype=int))
        .flatten()
        .tolist()
    )

    # This part marks the marking qubit if threshold is exceeded
    state_preparation.mcx(adder.qregs[1][threshold_qbts], marking_qubit)

    return state_preparation


state_preparation_circuit = state_preparation_func(
    probabilities, impact, threshold=12
)
state_preparation_circuit.decompose("adder").draw(output="mpl")

# Convert the state preparation circuit to some useful quantities
num_qubits_state_preparation = state_preparation_circuit.num_qubits
state_preparation_gate = state_preparation_circuit.to_gate()

# ## Some optimisations in the circuit
#
# (see [this blogpost](https://tinyurl.com/59r7htwc) for more info)

# +
local_simulator = Aer.get_backend("statevector_simulator")
basis_gates = local_simulator.configuration().basis_gates


def specific_adder():
    """
    Create optimised adder.

    A quantum circuit to create the specific adder that adds impacts of
    [0,1,4,8] on risk item 1-4.
    """
    weights = [0, 1, 4, 8]
    # Determine the number of state qubits
    N_state = len(weights)

    # Determine the number of sum qubits
    max_sum = sum(weights)
    N_sum = int(np.log2(max_sum)) + 1

    # Create the quantum registers
    state_reg = QuantumRegister(N_state, name="state")
    sum_reg = QuantumRegister(N_sum, name="sum")

    # Create adder circuit
    adder = QuantumCircuit(state_reg, sum_reg, name="adder")

    # Add 1 with qubit 2
    adder.mcx(
        control_qubits=[state_reg[2], *sum_reg[: N_sum - 1]],
        target_qubit=sum_reg[N_sum - 1],
    )
    adder.mcx(
        control_qubits=[state_reg[2], *sum_reg[: N_sum - 2]],
        target_qubit=sum_reg[N_sum - 2],
    )
    adder.mcx(
        control_qubits=[state_reg[2], *sum_reg[: N_sum - 3]],
        target_qubit=sum_reg[N_sum - 3],
    )
    adder.mcx(
        control_qubits=[state_reg[2], *sum_reg[: N_sum - 4]],
        target_qubit=sum_reg[N_sum - 4],
    )

    # Add 4 with qubit 1
    adder.mcx(
        control_qubits=[state_reg[1], *sum_reg[2 : N_sum - 1]],
        target_qubit=sum_reg[N_sum - 1],
    )
    adder.mcx(
        control_qubits=[state_reg[1], *sum_reg[2 : N_sum - 2]],
        target_qubit=sum_reg[N_sum - 2],
    )

    # Add 8 with qubit 0
    adder.mcx(
        control_qubits=[state_reg[0], *sum_reg[3 : N_sum - 1]],
        target_qubit=sum_reg[N_sum - 1],
    )

    return adder


# Create the specific adder
adder = specific_adder()

# Create state preparation circuit
state_preparation_circuit = state_preparation_func(
    probabilities, impact, threshold=12, adder=adder
)

state_preparation_circuit.draw(output="mpl")
# -

# ## Putting it together

# +
# Create the specific adder
adder = specific_adder()

# Create state preparation circuit
state_preparation_circuit = state_preparation_func(
    probabilities, impact, threshold=12, adder=adder
)

# Convert the state preparation circuit to some useful quantities
num_qubits_state_preparation = state_preparation_circuit.num_qubits
state_preparation_gate = state_preparation_circuit.to_gate()

# Create AE instances (oracle and grover operator)
state_preparation = QuantumCircuit(num_qubits_state_preparation)
state_preparation.append(state_preparation_circuit, state_preparation.qubits)

oracle = QuantumCircuit(num_qubits_state_preparation)
oracle.z(num_qubits_state_preparation - 1)

grover_opp = GroverOperator(oracle, state_preparation.decompose())

# +
problem = EstimationProblem(
    state_preparation=state_preparation,  # A operator
    grover_operator=grover_opp,  # Q operator
    objective_qubits=[
        num_qubits_state_preparation - 1
    ],  # the "good" state Psi1 is identified as measuring |1> in qubit 0
)

exact_p = Statevector(state_preparation).probabilities(
    qargs=[num_qubits_state_preparation - 1]
)[1]

print(f"Exact Probability we're trying to estimate: {exact_p:.5f}")

alpha = 0.01
epsilon_target = 1e-3


# -

# # Comparing Amplitude estimation algorithms


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

max_nb_qubits = 6
ae_nb_qubits = np.arange(1, max_nb_qubits + 1)

# +
# quantum sampler
shots_ae = 1

sampler_ae = Sampler(backend_options={"shots": shots_ae})
# -

# %%time
ae_results = []
for qbts in tqdm(ae_nb_qubits):
    # the number of evaluation qbts specifies circuit width and accuracy
    ae = AmplitudeEstimation(
        num_eval_qubits=qbts,
        sampler=sampler_ae,
    )
    ae_result = ae.estimate(problem)
    ae_result.confidence_interval = ae.compute_confidence_interval(
        ae_result, alpha
    )

    ae_results.append(ae_result)

print("Amplitude Estimation")
print(f"Exact probability: {exact_p:.4f}")
print(pretty_print_result(ae_results[-1]))

# ## Iterative AE

# +
# quantum sampler
shots_iae = 4

sampler_iae = Sampler(backend_options={"shots": shots_iae})
# -

iae = IterativeAmplitudeEstimation(
    epsilon_target=epsilon_target,  # target accuracy
    alpha=alpha,  # width of the confidence interval
    sampler=sampler_iae,
)

# %%time
iae_result = iae.estimate(problem)

print("Iterative Amplitude Estimation")
print(f"Exact probability: {exact_p:.4f}")
print(pretty_print_result(iae_result))

# ## Maximum Likelihood AE

max_power = 11
mlae_powers = 2 ** np.arange(0, max_power)

# quantum sampler
shots_mlae = 1

# +
# %%time
mlae_results = []
samplers_mlae = []

for i, power in tqdm(enumerate(mlae_powers), total=max_power):

    sampler_mlae = Sampler(backend_options={"shots": shots_mlae})

    mlae = MaximumLikelihoodAmplitudeEstimation(
        evaluation_schedule=i, sampler=sampler_mlae
    )

    mlae_result = mlae.estimate(problem)
    mlae_result.confidence_interval = mlae.compute_confidence_interval(
        mlae_result, alpha
    )

    mlae_results.append(mlae_result)

    samplers_mlae.append(sampler_mlae)
# -

print("Maximum Likelihood Amplitude Estimation")
print(f"Exact probability: {exact_p:.4f}")
print(pretty_print_result(mlae_results[-1]))

# ## Exqaliber AE

# +
# quantum sampler
shots_exae = 1

sampler_exae = Sampler(backend_options={"shots": shots_exae})
# -

exae = ExqaliberAmplitudeEstimation(
    sampler=sampler_exae,
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
x_ae = np.repeat((2 * ae_powers).cumsum() + 1, shots_ae)
y_1 = np.repeat(conf_intervals[:, 0], shots_ae)
y_2 = np.repeat(conf_intervals[:, 1], shots_ae)

axs[0, 0].fill_between(x_ae, y_1, y_2, alpha=0.5, label="Canonical AE")

# IAE
conf_intervals = np.array(iae_result.estimate_intervals)
iae_oracle_queries = shots_iae * (2 * np.array(iae_result.powers) + 1)
x_iae = iae_oracle_queries.cumsum()
y_1 = conf_intervals[:, 0]
y_2 = conf_intervals[:, 1]

axs[0, 1].fill_between(x_iae, y_1, y_2, alpha=0.5, label="Iterative AE")

# MLAE
conf_intervals = np.array(
    [result.confidence_interval for result in mlae_results]
)
mlae_oracle_queries = 2 * np.array(mlae_powers) + 1
x_mlae = np.repeat(mlae_oracle_queries * shots_mlae, shots_mlae)
y_1 = np.repeat(conf_intervals[:, 0], shots_mlae)
y_2 = np.repeat(conf_intervals[:, 1], shots_mlae)

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
y = np.repeat([result.estimation for result in ae_results], shots_ae)
axs[0, 0].plot(x_ae, y)

# IAE
y = np.array(iae_result.estimate_intervals).mean(axis=1)
axs[0, 1].plot(x_iae, y)

# MLAE
y = np.repeat([result.estimation for result in mlae_results], shots_mlae)
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
fig.suptitle("Estimation vs. oracle calls (risk modelling)")
plt.savefig("results/estimation-oracle-calls.png")

plt.tight_layout()

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
x_ae = np.repeat((2 * ae_powers).cumsum() + 1, shots_ae)
y_ae = np.repeat(ae_width_conf, shots_ae)
ax.plot(x_ae, y_ae, label="Canonical AE")


# IAE
conf_intervals = np.array(iae_result.estimate_intervals)
iae_oracle_queries = 2 * np.array(iae_result.powers) + 1
iae_width_conf = conf_intervals[:, 1] - conf_intervals[:, 0]
x_iae = np.repeat(iae_oracle_queries, shots_iae).cumsum()
y_iae = np.repeat(iae_width_conf, shots_iae)

ax.plot(x_iae, y_iae, label="Iterative AE")

# MLAE
conf_intervals = np.array(
    [result.confidence_interval for result in mlae_results]
)
mlae_oracle_queries = 2 * np.array(mlae_powers) + 1
mlae_width_conf = conf_intervals[:, 1] - conf_intervals[:, 0]
x_mlae = np.repeat(mlae_oracle_queries * shots_mlae, shots_mlae)
y_mlae = np.repeat(mlae_width_conf, shots_mlae)

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
ax.set_title("Width of confidence interval vs. oracle calls (risk model)")
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
ax.set_title("Power of grover operator (risk model)")
ax.legend()

plt.savefig("results/powers-comparison.png")

plt.show()
# -
# ## Depth circuit used

# +
fig, ax = plt.subplots()

# CAE
circuits_ae = list(sampler_ae._transpiled_circuits.values())
y_ae = [qc.depth() for qc in circuits_ae]

conf_intervals = np.array(
    [result.confidence_interval for result in ae_results]
)
ae_width_conf = conf_intervals[:, 1] - conf_intervals[:, 0]
x_ae = ae_width_conf

ax.scatter(1 / x_ae, y_ae, label="Canonical AE", marker="x")

# IAE
circuits_iae = list(sampler_iae._transpiled_circuits.values())
y_iae = [qc.depth() for qc in circuits_iae]

conf_intervals = np.array(iae_result.estimate_intervals[1:])
iae_width_conf = conf_intervals[:, 1] - conf_intervals[:, 0]
x_iae = iae_width_conf

ax.scatter(1 / x_iae, y_iae, label="Iterative AE", marker="x")

# MLAE
circuits_mlae = [
    list(sampler_mlae._transpiled_circuits.values())
    for sampler_mlae in samplers_mlae
]
y_mlae = [qcs[-1].depth() for qcs in circuits_mlae]

conf_intervals = np.array(
    [result.confidence_interval for result in mlae_results]
)
mlae_width_conf = conf_intervals[:, 1] - conf_intervals[:, 0]
x_mlae = mlae_width_conf

ax.scatter(1 / x_mlae, y_mlae, label="Maximum Likelihood AE", marker="x")

# EXAE
circuits_exae = list(sampler_exae._transpiled_circuits.values())
y_exae = [qc.depth() for qc in circuits_exae]

conf_intervals = np.array(exae_result.estimate_intervals[1:])
exae_width_conf = conf_intervals[:, 1] - conf_intervals[:, 0]
x_exae = exae_width_conf

ax.scatter(1 / x_exae, y_exae, label="Exqaliber AE", marker="x")

# Axes
ax.set_xscale("log")
ax.set_xlabel(r"$(1/\epsilon)$")
ax.set_xlim(1, 1 / (epsilon_target / 3))
ax.set_yscale("log")
ax.set_ylabel("Max depth circuit used")

# Title
ax.set_title("accuracy vs. circuit depth (risk model)")
ax.legend()

# plt.savefig("results/width-confinterval-shots.png")

plt.show()
