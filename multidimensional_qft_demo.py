from __future__ import annotations

import argparse

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from multidimensional_qft import (
    build_multidimensional_qft,
    dimension_qubit_widths,
    expected_multidimensional_qft_state,
    padded_shape,
    prepare_multidimensional_input,
    validate_shape,
)
from qft_sampler_utils import (
    build_measured_qft_circuit,
    sample_aer_counts,
    sample_noisy_aer_counts,
    select_fake_backend,
    top_outcomes,
    total_variation_distance,
)
from qft_visualization_utils import prepare_circuit_for_display


# Build a simple deterministic array for demonstration.
def build_sample_array(shape: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod(shape))
    return np.arange(1, size + 1, dtype=float).reshape(shape, order="F")


# Print a short summary of the most likely outputs.
def describe_counts(label: str, counts: dict[str, int]) -> None:
    print(f"\n{label}:")
    for bitstring, count in top_outcomes(counts):
        print(f"  {bitstring}: {count}")


# Print simple circuit metrics.
def describe_circuit(label: str, circuit: QuantumCircuit) -> None:
    print(f"\n{label}:")
    print(prepare_circuit_for_display(circuit).draw(output="text", fold=160, idle_wires=False))
    print(f"Depth: {circuit.depth()}")
    print(f"Operations: {dict(circuit.count_ops())}")


# Run the multidimensional QFT demo from the command line.
def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multidimensional QFT demo.")
    parser.add_argument("--shape", type=int, nargs="+", default=[4, 4], help="Dimension sizes. Non-power-of-two sizes are padded with zeros.")
    parser.add_argument("--shots", type=int, default=1024, help="Number of Aer shots to use.")
    parser.add_argument(
        "--method",
        choices=("standard", "recursive", "both"),
        default="both",
        help="Which multidimensional QFT construction to show.",
    )
    args = parser.parse_args()

    shape = validate_shape(tuple(args.shape))
    padded = padded_shape(shape)
    widths = dimension_qubit_widths(shape)
    total_qubits = sum(widths)
    sample_array = build_sample_array(shape)
    padded_array, amplitudes = prepare_multidimensional_input(sample_array)
    expected_state = expected_multidimensional_qft_state(sample_array)

    standard_qft = build_multidimensional_qft(shape, method="standard")
    recursive_qft = build_multidimensional_qft(shape, method="recursive")

    standard_input = QuantumCircuit(total_qubits)
    standard_input.initialize(amplitudes, range(total_qubits))
    standard_input.compose(standard_qft, inplace=True)

    recursive_input = QuantumCircuit(total_qubits)
    recursive_input.initialize(amplitudes, range(total_qubits))
    recursive_input.compose(recursive_qft, inplace=True)

    standard_output = Statevector.from_instruction(standard_input).data
    recursive_output = Statevector.from_instruction(recursive_input).data

    print(f"Original shape: {shape}")
    print(f"Padded shape: {padded}")
    print(f"Dimension qubits: {widths}")
    print(f"Total qubits: {total_qubits}")
    if padded != shape:
        print("\nThe input array was zero-padded to the next power of two in each dimension.")
    print("\nInput array after padding:")
    print(np.array2string(padded_array, precision=4, suppress_small=True))
    print("\nFlattened input amplitudes:")
    print(np.array2string(amplitudes, precision=4, suppress_small=True))
    print("\nExpected multidimensional DFT amplitudes:")
    print(np.array2string(expected_state, precision=4, suppress_small=True))
    print("\nMultidimensional QFT amplitudes from the standard method:")
    print(np.array2string(standard_output, precision=4, suppress_small=True))
    print("\nMultidimensional QFT amplitudes from the recursive method:")
    print(np.array2string(recursive_output, precision=4, suppress_small=True))

    backend = select_fake_backend(total_qubits)
    pass_manager = generate_preset_pass_manager(optimization_level=2, backend=backend)
    print(f"\nUsing fake backend: {backend.name}")

    standard_measured = build_measured_qft_circuit(amplitudes, standard_qft)
    recursive_measured = build_measured_qft_circuit(amplitudes, recursive_qft)

    standard_aer_counts = sample_aer_counts(standard_measured, shots=args.shots)
    recursive_aer_counts = sample_aer_counts(recursive_measured, shots=args.shots)
    standard_noisy_counts = sample_noisy_aer_counts(standard_measured, backend=backend, shots=args.shots)
    recursive_noisy_counts = sample_noisy_aer_counts(recursive_measured, backend=backend, shots=args.shots)

    describe_counts("Ideal Aer counts for the standard multidimensional QFT", standard_aer_counts)
    describe_counts("Ideal Aer counts for the recursive multidimensional QFT", recursive_aer_counts)
    describe_counts("Noisy Aer counts for the standard multidimensional QFT", standard_noisy_counts)
    describe_counts("Noisy Aer counts for the recursive multidimensional QFT", recursive_noisy_counts)

    print(
        "\nAer comparison:"
        f"\n  Ideal Aer TVD between methods: {total_variation_distance(standard_aer_counts, recursive_aer_counts):.4f}"
        f"\n  Standard ideal vs noisy TVD: {total_variation_distance(standard_aer_counts, standard_noisy_counts):.4f}"
        f"\n  Recursive ideal vs noisy TVD: {total_variation_distance(recursive_aer_counts, recursive_noisy_counts):.4f}"
    )

    if args.method == "standard":
        describe_circuit("Standard multidimensional QFT", standard_qft)
        describe_circuit("Standard multidimensional QFT transpiled", pass_manager.run(standard_qft))
    elif args.method == "recursive":
        describe_circuit("Recursive multidimensional QFT", recursive_qft)
        describe_circuit("Recursive multidimensional QFT transpiled", pass_manager.run(recursive_qft))
    else:
        describe_circuit("Standard multidimensional QFT", standard_qft)
        describe_circuit("Standard multidimensional QFT transpiled", pass_manager.run(standard_qft))
        describe_circuit("Recursive multidimensional QFT", recursive_qft)
        describe_circuit("Recursive multidimensional QFT transpiled", pass_manager.run(recursive_qft))


if __name__ == "__main__":
    main()
