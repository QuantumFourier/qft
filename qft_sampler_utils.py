from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile


# Build the same sample input state used across the QFT demos.
def build_sample_amplitudes(num_qubits: int) -> np.ndarray:
    amplitudes = np.arange(1, 2**num_qubits + 1, dtype=float)
    return amplitudes / np.linalg.norm(amplitudes)


# Attach measurements so the simulator can produce output counts.
def build_measured_qft_circuit(amplitudes: np.ndarray, qft_circuit: QuantumCircuit) -> QuantumCircuit:
    num_qubits = qft_circuit.num_qubits
    circuit = QuantumCircuit(num_qubits)
    circuit.initialize(amplitudes, range(num_qubits))
    circuit.compose(qft_circuit, inplace=True)
    circuit.measure_all()
    return circuit


# Load Aer only when it is actually requested.
def require_aer():
    try:
        from qiskit_aer import AerSimulator
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qiskit_aer is not installed. Install it with `pip install qiskit-aer`."
        ) from exc
    return AerSimulator


# Pick a fake backend that is large enough for the requested circuit.
def select_fake_backend(num_qubits: int):
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeKolkataV2, FakeManilaV2

    candidates = [FakeManilaV2(), FakeKolkataV2(), FakeBrisbane()]
    for backend in candidates:
        if backend.num_qubits >= num_qubits:
            return backend

    raise ValueError(
        f"No available fake backend has enough qubits for a {num_qubits}-qubit circuit."
    )


# Run an ideal Aer simulation for a measured QFT circuit.
def sample_aer_counts(circuit: QuantumCircuit, shots: int, method: str = "automatic") -> dict[str, int]:
    AerSimulator = require_aer()
    simulator = AerSimulator() if method == "automatic" else AerSimulator(method=method)
    transpiled = transpile(circuit, simulator)
    result = simulator.run(transpiled, shots=shots).result()
    return dict(result.get_counts(0))


# Run a noisy Aer simulation using a fake backend as the noise source.
def sample_noisy_aer_counts(circuit: QuantumCircuit, backend, shots: int) -> dict[str, int]:
    AerSimulator = require_aer()
    simulator = AerSimulator.from_backend(backend)
    transpiled = transpile(circuit, simulator)
    result = simulator.run(transpiled, shots=shots).result()
    return dict(result.get_counts(0))


# Turn raw counts into probabilities for easier comparison.
def counts_to_probabilities(counts: dict[str, int]) -> dict[str, float]:
    total_shots = sum(counts.values())
    if total_shots == 0:
        return {}
    return {bitstring: count / total_shots for bitstring, count in counts.items()}


# Keep only the most likely bitstrings in sorted order.
def top_outcomes(counts: dict[str, int], limit: int = 8) -> list[tuple[str, int]]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:limit]


# Compute a simple distance between two sampled distributions.
def total_variation_distance(first: dict[str, int], second: dict[str, int]) -> float:
    first_probs = counts_to_probabilities(first)
    second_probs = counts_to_probabilities(second)
    support = set(first_probs) | set(second_probs)
    return 0.5 * sum(abs(first_probs.get(key, 0.0) - second_probs.get(key, 0.0)) for key in support)


# Create a compact sampler summary for reports and terminal output.
def counts_summary(counts: dict[str, int], limit: int = 8) -> dict[str, Any]:
    return {
        "shots": int(sum(counts.values())),
        "counts": counts,
        "top_outcomes": top_outcomes(counts, limit=limit),
        "probabilities": counts_to_probabilities(counts),
    }
