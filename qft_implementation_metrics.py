from __future__ import annotations

import argparse
import json
from math import prod
from pathlib import Path
from time import perf_counter

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from forward_qft import build_forward_qft, build_recursive_forward_qft


# Build one QFT circuit and record how long construction takes.
def build_with_timing(builder, num_qubits: int) -> tuple[QuantumCircuit, float]:
    start = perf_counter()
    circuit = builder(num_qubits)
    elapsed = perf_counter() - start
    return circuit, elapsed


# Run transpilation and record how long the pass manager takes.
def transpile_with_timing(circuit: QuantumCircuit, pass_manager) -> tuple[QuantumCircuit, float]:
    start = perf_counter()
    transpiled = pass_manager.run(circuit)
    elapsed = perf_counter() - start
    return transpiled, elapsed


# Convert a circuit instruction into backend qubit indices.
def qubit_indices(circuit: QuantumCircuit, instruction) -> list[int]:
    return [circuit.find_bit(qubit).index for qubit in instruction.qubits]


# Estimate duration and error from the backend calibration data.
def estimate_backend_costs(circuit: QuantumCircuit, backend) -> dict[str, float | int]:
    properties = backend.properties()
    total_duration = 0.0
    total_error = 0.0
    cnot_error = 0.0
    cnot_count = 0
    gate_errors: list[float] = []
    missing_properties = 0

    for instruction in circuit.data:
        gate_name = instruction.operation.name
        indices = qubit_indices(circuit, instruction)

        try:
            gate_error = float(properties.gate_error(gate_name, indices))
            gate_duration = float(properties.gate_length(gate_name, indices))
        except Exception:
            missing_properties += 1
            continue

        total_error += gate_error
        total_duration += gate_duration
        gate_errors.append(gate_error)

        if gate_name == "cx":
            cnot_count += 1
            cnot_error += gate_error

    success_probability = prod(1.0 - error for error in gate_errors) if gate_errors else 1.0

    return {
        "estimated_duration_seconds": total_duration,
        "estimated_total_error": total_error,
        "estimated_cnot_error": cnot_error,
        "estimated_success_probability": success_probability,
        "transpiled_cnot_count": cnot_count,
        "operations_missing_backend_properties": missing_properties,
    }


# Collect one set of implementation properties for a single method.
def collect_method_metrics(label: str, builder, num_qubits: int, backend, pass_manager) -> dict:
    original_circuit, build_time = build_with_timing(builder, num_qubits)
    transpiled_circuit, transpile_time = transpile_with_timing(original_circuit, pass_manager)
    backend_costs = estimate_backend_costs(transpiled_circuit, backend)

    return {
        "method": label,
        "build_time_seconds": build_time,
        "transpile_time_seconds": transpile_time,
        "original_depth": original_circuit.depth(),
        "original_gate_count": original_circuit.size(),
        "original_gate_breakdown": dict(original_circuit.count_ops()),
        "transpiled_depth": transpiled_circuit.depth(),
        "transpiled_gate_count": transpiled_circuit.size(),
        "transpiled_gate_breakdown": dict(transpiled_circuit.count_ops()),
        **backend_costs,
    }


# Print a compact comparison table for the two QFT methods.
def print_summary(metrics: list[dict]) -> None:
    print(
        f"{'Method':<22}{'Build(s)':>12}{'Transpile(s)':>14}"
        f"{'CNOTs':>10}{'Gates':>10}{'Error':>14}"
    )
    for item in metrics:
        print(
            f"{item['method']:<22}{item['build_time_seconds']:>12.6f}"
            f"{item['transpile_time_seconds']:>14.6f}"
            f"{item['transpiled_cnot_count']:>10}"
            f"{item['transpiled_gate_count']:>10}"
            f"{item['estimated_total_error']:>14.6f}"
        )


# Generate and save the implementation report.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record implementation properties for the standard QFT method and recursive QFT."
    )
    parser.add_argument("--qubits", type=int, default=3, help="Number of qubits in the QFT circuits.")
    parser.add_argument(
        "--output",
        default="qft_implementation_metrics.json",
        help="Output JSON file for the recorded metrics.",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=2,
        choices=(0, 1, 2, 3),
        help="Qiskit transpilation optimization level.",
    )
    args = parser.parse_args()

    backend = FakeManilaV2()
    pass_manager = generate_preset_pass_manager(
        optimization_level=args.optimization_level,
        backend=backend,
    )

    metrics = [
        collect_method_metrics(
            "Standard QFT method",
            build_forward_qft,
            args.qubits,
            backend,
            pass_manager,
        ),
        collect_method_metrics(
            "Recursive QFT",
            build_recursive_forward_qft,
            args.qubits,
            backend,
            pass_manager,
        ),
    ]

    report = {
        "backend_name": backend.name,
        "num_qubits": args.qubits,
        "optimization_level": args.optimization_level,
        "metrics": metrics,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    output_path.write_text(json.dumps(report, indent=2))

    print_summary(metrics)
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
