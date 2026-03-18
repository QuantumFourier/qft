from __future__ import annotations

import argparse

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from forward_qft import build_forward_qft, build_recursive_forward_qft, qft_on_amplitudes


# Build a sample input state from a chosen amplitude vector.
def build_input_circuit(amplitudes: np.ndarray) -> QuantumCircuit:
    num_qubits = int(np.log2(amplitudes.size))
    circuit = QuantumCircuit(num_qubits)
    circuit.initialize(amplitudes, range(num_qubits))
    return circuit


# Print a circuit and a few simple metrics.
def describe_circuit(label: str, circuit: QuantumCircuit) -> None:
    print(f"\n{label}:")
    print(circuit.draw(output="text"))
    print(f"Depth: {circuit.depth()}")
    print(f"Operations: {dict(circuit.count_ops())}")


# Transpile one circuit and print before-and-after details.
def transpile_and_report(
    label: str,
    circuit: QuantumCircuit,
    pass_manager,
) -> QuantumCircuit:
    transpiled = pass_manager.run(circuit)
    describe_circuit(f"{label} (original)", circuit)
    describe_circuit(f"{label} (transpiled)", transpiled)
    return transpiled


# Compare the transpilation cost of both QFT constructions.
def compare_transpilation_costs(
    standard_qft: QuantumCircuit,
    recursive_qft: QuantumCircuit,
    pass_manager,
) -> None:
    standard_transpiled = transpile_and_report("Standard QFT", standard_qft, pass_manager)
    recursive_transpiled = transpile_and_report("Recursive QFT", recursive_qft, pass_manager)

    print("\nTranspilation cost comparison:")
    print(
        f"{'Method':<18}{'Orig depth':>12}{'Trans depth':>14}"
        f"{'Orig size':>12}{'Trans size':>12}"
    )
    print(
        f"{'Standard':<18}{standard_qft.depth():>12}{standard_transpiled.depth():>14}"
        f"{standard_qft.size():>12}{standard_transpiled.size():>12}"
    )
    print(
        f"{'Recursive':<18}{recursive_qft.depth():>12}{recursive_transpiled.depth():>14}"
        f"{recursive_qft.size():>12}{recursive_transpiled.size():>12}"
    )


# Run the full demo from the command line.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build and transpile standard QFT method and recursive QFT circuits.")
    parser.add_argument("--qubits", type=int, default=3, help="Number of qubits in the QFT circuit.")
    parser.add_argument(
        "--method",
        choices=("standard", "recursive", "both"),
        default="both",
        help="Which QFT construction to display and transpile.",
    )
    args = parser.parse_args()

    num_qubits = args.qubits
    sample_amplitudes = np.arange(1, 2**num_qubits + 1, dtype=float)
    sample_amplitudes = sample_amplitudes / np.linalg.norm(sample_amplitudes)

    standard_qft = build_forward_qft(num_qubits)
    recursive_qft = build_recursive_forward_qft(num_qubits)

    reference = QuantumCircuit(num_qubits)
    reference.append(QFTGate(num_qubits), range(num_qubits))

    print(f"Standard circuit matches Qiskit's QFTGate: {Operator(standard_qft).equiv(Operator(reference))}")
    print(f"Recursive circuit matches Qiskit's QFTGate: {Operator(recursive_qft).equiv(Operator(reference))}")

    transformed = qft_on_amplitudes(sample_amplitudes)
    standard_input = build_input_circuit(sample_amplitudes)
    standard_input.compose(standard_qft, inplace=True)
    recursive_input = build_input_circuit(sample_amplitudes)
    recursive_input.compose(recursive_qft, inplace=True)

    standard_output_state = Statevector.from_instruction(standard_input)
    recursive_output_state = Statevector.from_instruction(recursive_input)
    print("\nInput amplitudes:")
    print(np.array2string(sample_amplitudes, precision=4, suppress_small=True))
    print("\nQFT amplitudes from the DFT formula:")
    print(np.array2string(transformed, precision=4, suppress_small=True))
    print("\nQFT amplitudes from the standard QFT method:")
    print(np.array2string(standard_output_state.data, precision=4, suppress_small=True))
    print("\nQFT amplitudes from the recursive circuit:")
    print(np.array2string(recursive_output_state.data, precision=4, suppress_small=True))

    backend = FakeManilaV2()
    pass_manager = generate_preset_pass_manager(optimization_level=2, backend=backend)

    if args.method == "standard":
        transpile_and_report("Standard QFT", standard_qft, pass_manager)
    elif args.method == "recursive":
        transpile_and_report("Recursive QFT", recursive_qft, pass_manager)
    else:
        compare_transpilation_costs(standard_qft, recursive_qft, pass_manager)


if __name__ == "__main__":
    main()
