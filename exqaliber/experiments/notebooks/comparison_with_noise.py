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

"""Compare different amplitude estimation algorithms."""

# # Noisy amplitude estimation

# ### Set-up

# +

import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms import AmplitudeEstimation, EstimationProblem
from qiskit.circuit import EquivalenceLibrary, Instruction
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    BasisTranslator,
    Decompose,
    UnrollCustomDefinitions,
)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import LocalNoisePass, depolarizing_error

# from exqaliber.exqaliber_amplitude_estimation import (
#     ExqaliberAmplitudeEstimation,
# )


np.random.seed(1)


# +
class ControlledBernoulliA(QuantumCircuit):
    """A circuit representing the Bernoulli A operator."""

    def __init__(self, probability):
        super().__init__(2, name="A")  # circuit on 2 qubits

        theta_p = 2 * np.arcsin(np.sqrt(2 * probability))
        self.h(0)
        self.cry(theta_p, 0, 1)


class CustomGroverOperator(QuantumCircuit):
    """A custom Grover Operator."""

    def __init__(self, oracle, A):
        super().__init__(A.num_qubits, name="Q")
        self.append(A.inverse().to_gate(), range(A.num_qubits))
        self.append(oracle.to_gate(), range(A.num_qubits))
        self.append(A.to_gate(), range(A.num_qubits))

    def to_gate(self, parameter_map=None, label=None):
        """Create a gate from instruction."""
        return super().to_gate(parameter_map, label)

    def power(self, power):
        """Raise the operator to a power."""
        power_circuit = self.copy_empty_like(name=f"Q^{power}")
        for i in range(power):
            power_circuit.append(self, range(self.num_qubits))

        return power_circuit.decompose()


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


p = 0.2
A = ControlledBernoulliA(p)
obj = [1]
exact_p = Statevector(A).probabilities(qargs=obj)[1]

oracle = QuantumCircuit(A.num_qubits, name="mark")
oracle.z(obj)
Q = CustomGroverOperator(oracle, A)

problem = EstimationProblem(
    state_preparation=A.to_gate(),  # A operator
    grover_operator=Q,  # Q operator
    objective_qubits=obj,
)

print(f"Exact Probability we're trying to estimate: {exact_p:.5f}")

alpha = 0.01
epsilon_target = 1e-4
# -

# ### Adding noise

# +
shots_ae = 1024
sampler_ae = Sampler(options={"shots": shots_ae})

noise_level = 0.01
# -

ae = AmplitudeEstimation(
    num_eval_qubits=4,
    sampler=sampler_ae,
)

qc = ae.construct_circuit(problem)
qc.decompose("QPE").draw("mpl")

# +
basis_gates = AerSimulator().configuration().basis_gates

# intialize pass mananger
# stg_pm = StagedPassManager(stages=['QPE', 'powers', 'controlled'])
pm = PassManager()

# passes
decompose_QPE = Decompose(["QPE"])
pm.append(decompose_QPE)

equiv_lib_cQ = EquivalenceLibrary()
controlled_Q = Q.control(1)
basis_gates_cQ = basis_gates[:]
basis_gates_cQ.append("cQ")
basis_gates_cQ.append("A")

for inst in pm.run(qc):
    name = inst.operation.name
    if "Q^" in name:
        power = name[name.find("^") + 1 :]
        power_circuit = controlled_Q.power(int(power))
        equiv_lib_cQ.add_equivalence(inst.operation, power_circuit.decompose())

unroller_cQ = UnrollCustomDefinitions(equiv_lib_cQ, basis_gates=basis_gates_cQ)
pm.append(unroller_cQ)

basis_translator_cQ = BasisTranslator(
    equiv_lib_cQ, target_basis=basis_gates_cQ
)
pm.append(basis_translator_cQ)

equiv_lib_A = EquivalenceLibrary()

c_Q = QuantumCircuit(A.num_qubits + 1)
for inst in Q:
    c_Q.append(inst.operation.control(1), range(A.num_qubits + 1))

for inst in pm.run(qc):
    name = inst.operation.name
    if name == "cQ":
        equiv_lib_A.add_equivalence(inst.operation, c_Q)
        break

basis_gates_A = basis_gates[:]
basis_gates_A.append("A")
basis_gates_A.append("cA")
basis_gates_A.append("cmark")
basis_gates_A.append("cA_dg")

unroller_A = UnrollCustomDefinitions(equiv_lib_A, basis_gates=basis_gates_A)
pm.append(unroller_A)

basis_translator_A = BasisTranslator(equiv_lib_A, target_basis=basis_gates_A)
pm.append(basis_translator_A)

# create all-qubit error


def noise_func(inst: Instruction, qubits):
    """Add noise after an oracle call."""
    if "A" not in inst.name:
        return None
    error = depolarizing_error(noise_level, inst.num_qubits)
    error_qc = QuantumCircuit(inst.num_qubits, inst.num_clbits)
    # error_qc.append(inst, qargs=range(inst.num_qubits),
    # cargs=range(inst.num_clbits))
    error_qc.append(
        error, qargs=range(inst.num_qubits), cargs=range(inst.num_clbits)
    )
    return error  # .to_instruction()


noise_pass = LocalNoisePass(noise_func, method="append")
pm.append(noise_pass)
# -

pm.run(qc).draw("mpl")

# +
# quantum sampler
shots_ae = 1024

sampler_ae = Sampler(options={"shots": shots_ae})
# -

ae = AmplitudeEstimation(
    num_eval_qubits=4,
    sampler=sampler_ae,
)
ae_result = ae.estimate(problem)
# ae_noisy_result = ae.estimate(noisy_problem)

print("Amplitude Estimation")
print(f"Exact probability: {exact_p:.4f}")
print(pretty_print_result(ae_result))
print("Noisy results:")
# print(pretty_print_result(ae_noisy_result))
