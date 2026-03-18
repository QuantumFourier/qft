from __future__ import annotations

from math import pi, sqrt

import numpy as np
from qiskit import QuantumCircuit


def dft_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    """Return the discrete Fourier transform used by the forward QFT."""
    values = np.asarray(amplitudes, dtype=complex)
    size = values.size

    if size == 0:
        raise ValueError("The amplitude vector must not be empty.")

    indices = np.arange(size)
    phase_matrix = np.exp(2j * pi * np.outer(indices, indices) / size)
    return phase_matrix @ values / sqrt(size)


def build_forward_qft(num_qubits: int, do_swaps: bool = True) -> QuantumCircuit:
    """Build the standard forward-QFT circuit using H, controlled-phase, and SWAP gates."""
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1.")

    circuit = QuantumCircuit(num_qubits, name=f"QFT_{num_qubits}")

    for target in reversed(range(num_qubits)):
        circuit.h(target)
        for control in reversed(range(target)):
            angle = pi / (1 << (target - control))
            circuit.cp(angle, control, target)

    if do_swaps:
        for left in range(num_qubits // 2):
            circuit.swap(left, num_qubits - left - 1)

    return circuit


def build_recursive_forward_qft(num_qubits: int) -> QuantumCircuit:
    """Build the forward QFT using the recursive construction."""
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1.")

    circuit = QuantumCircuit(num_qubits, name=f"RecursiveQFT_{num_qubits}")
    _append_recursive_qft(circuit, list(range(num_qubits)))
    return circuit


def _append_recursive_qft(circuit: QuantumCircuit, qubits: list[int]) -> None:
    if len(qubits) == 1:
        circuit.h(qubits[0])
        return

    _append_recursive_qft(circuit, qubits[1:])

    target = qubits[0]
    width = len(qubits)
    for offset, control in enumerate(qubits[1:]):
        angle = pi / (1 << (width - offset - 1))
        circuit.cp(angle, control, target)

    circuit.h(target)

    # This adjacent-swap ladder performs the final qubit reordering from the recursive formula.
    for index in range(width - 1):
        circuit.swap(qubits[index], qubits[index + 1])


def qft_on_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    """Apply the forward-QFT amplitude map to a state vector."""
    values = np.asarray(amplitudes, dtype=complex)
    size = values.size

    if size == 0 or size & (size - 1):
        raise ValueError("The number of amplitudes must be a non-zero power of 2.")

    norm = np.linalg.norm(values)
    if norm == 0:
        raise ValueError("The amplitude vector must not be the zero vector.")

    normalized = values / norm
    return dft_amplitudes(normalized)
