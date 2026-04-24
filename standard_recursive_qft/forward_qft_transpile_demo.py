from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

PARENT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PARENT_DIR / "src"
for candidate in (PARENT_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from qft.standard_recursive_qft import build_recursive_qft, build_standard_qft, qft_on_amplitudes
from qft_sampler_utils import (
    build_measured_qft_circuit,
    build_sample_amplitudes,
    sample_aer_counts,
    sample_noisy_aer_counts,
    select_fake_backend,
    top_outcomes,
    total_variation_distance,
)
from qft_visualization_utils import prepare_circuit_for_display


# Build a sample input state from a chosen amplitude vector.
def build_input_circuit(amplitudes: np.ndarray) -> QuantumCircuit:
    num_qubits = int(np.log2(amplitudes.size))
    circuit = QuantumCircuit(num_qubits)
    circuit.initialize(amplitudes, range(num_qubits))
    return circuit


# Print a circuit and a few simple metrics.
def describe_circuit(label: str, circuit: QuantumCircuit) -> None:
    print(f"\n{label}:")
    print(prepare_circuit_for_display(circuit).draw(output="text", fold=160, idle_wires=False))
    print(f"Depth: {circuit.depth()}")
    print(f"Operations: {dict(circuit.count_ops())}")


# Print a short counts summary with the most likely results.
def describe_counts(label: str, counts: dict[str, int]) -> None:
    print(f"\n{label}:")
    for bitstring, count in top_outcomes(counts):
        print(f"  {bitstring}: {count}")


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


# Run both QFT methods on Aer and compare the measured output distributions.
def run_aer_comparisons(
    amplitudes: np.ndarray,
    standard_qft: QuantumCircuit,
    recursive_qft: QuantumCircuit,
    backend,
    shots: int,
) -> None:
    standard_measured = build_measured_qft_circuit(amplitudes, standard_qft)
    recursive_measured = build_measured_qft_circuit(amplitudes, recursive_qft)

    try:
        standard_aer_counts = sample_aer_counts(standard_measured, shots=shots)
        recursive_aer_counts = sample_aer_counts(recursive_measured, shots=shots)
        standard_noisy_aer_counts = sample_noisy_aer_counts(standard_measured, backend=backend, shots=shots)
        recursive_noisy_aer_counts = sample_noisy_aer_counts(recursive_measured, backend=backend, shots=shots)

        describe_counts("Ideal Aer counts for the standard QFT method", standard_aer_counts)
        describe_counts("Ideal Aer counts for the recursive QFT", recursive_aer_counts)
        describe_counts("Noisy Aer counts for the standard QFT method", standard_noisy_aer_counts)
        describe_counts("Noisy Aer counts for the recursive QFT", recursive_noisy_aer_counts)

        print(
            "\nAer comparison:"
            f"\n  Ideal Aer TVD between methods: {total_variation_distance(standard_aer_counts, recursive_aer_counts):.4f}"
            f"\n  Standard method ideal vs noisy Aer TVD: {total_variation_distance(standard_aer_counts, standard_noisy_aer_counts):.4f}"
            f"\n  Recursive QFT ideal vs noisy Aer TVD: {total_variation_distance(recursive_aer_counts, recursive_noisy_aer_counts):.4f}"
            f"\n  Noisy Aer TVD between methods: {total_variation_distance(standard_noisy_aer_counts, recursive_noisy_aer_counts):.4f}"
        )
    except RuntimeError as exc:
        print(f"\nAer simulator unavailable: {exc}")


# Run the full demo from the command line.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build and transpile standard QFT method and recursive QFT circuits.")
    parser.add_argument("--qubits", type=int, default=3, help="Number of qubits in the QFT circuit.")
    parser.add_argument("--shots", type=int, default=2048, help="Number of Aer shots to use.")
    parser.add_argument(
        "--method",
        choices=("standard", "recursive", "both"),
        default="both",
        help="Which QFT construction to display and transpile.",
    )
    args = parser.parse_args()

    num_qubits = args.qubits
    sample_amplitudes = build_sample_amplitudes(num_qubits)

    standard_qft = build_standard_qft(num_qubits)
    recursive_qft = build_recursive_qft(num_qubits)

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

    backend = select_fake_backend(num_qubits)
    pass_manager = generate_preset_pass_manager(optimization_level=2, backend=backend)
    print(f"\nUsing fake backend: {backend.name}")
    run_aer_comparisons(
        sample_amplitudes,
        standard_qft,
        recursive_qft,
        backend,
        shots=args.shots,
    )

    if args.method == "standard":
        transpile_and_report("Standard QFT", standard_qft, pass_manager)
    elif args.method == "recursive":
        transpile_and_report("Recursive QFT", recursive_qft, pass_manager)
    else:
        compare_transpilation_costs(standard_qft, recursive_qft, pass_manager)


if __name__ == "__main__":
    main()
