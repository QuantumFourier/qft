from __future__ import annotations

from math import pi, sqrt

import numpy as np
from qiskit import QuantumCircuit


def _compose_stage_circuits(
    num_qubits: int,
    stages: list[tuple[str, QuantumCircuit]],
    *,
    name: str,
    show_barriers: bool = False,
) -> QuantumCircuit:
    """Compose labeled stages into one circuit, optionally separating them with barriers."""
    circuit = QuantumCircuit(num_qubits, name=name)

    for index, (_, stage) in enumerate(stages):
        circuit.compose(stage, inplace=True)
        if show_barriers and index < len(stages) - 1:
            circuit.barrier()

    return circuit


# Direct amplitude-based calculation for the transform.
def dft_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    """Return the discrete Fourier transform used by the standard QFT method."""
    values = np.asarray(amplitudes, dtype=complex)
    size = values.size

    if size == 0:
        raise ValueError("The amplitude vector must not be empty.")

    indices = np.arange(size)
    phase_matrix = np.exp(2j * pi * np.outer(indices, indices) / size)
    return phase_matrix @ values / sqrt(size)


# Standard QFT method built from H, controlled-phase, and SWAP gates.
def build_standard_qft(
    num_qubits: int,
    do_swaps: bool = True,
    recursive: bool = False,
) -> QuantumCircuit:
    """Build either the standard or recursive QFT circuit.

    When ``recursive`` is true, this function delegates to
    :func:`build_recursive_qft`. In that mode, ``do_swaps`` is ignored because
    the recursive construction includes its own final qubit reordering.
    """
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative.")

    if recursive:
        return build_recursive_qft(num_qubits)

    stages = _standard_qft_stage_circuits(num_qubits, do_swaps=do_swaps)
    return _compose_stage_circuits(
        num_qubits,
        stages,
        name=f"QFT_{num_qubits}",
    )


def _standard_qft_stage_circuits(
    num_qubits: int,
    do_swaps: bool = True,
) -> list[tuple[str, QuantumCircuit]]:
    """Build the standard QFT as labeled stage circuits."""
    stages: list[tuple[str, QuantumCircuit]] = []

    for target in reversed(range(num_qubits)):
        stage = QuantumCircuit(num_qubits, name=f"LayerQ{target}")
        stage.h(target)
        for control in reversed(range(target)):
            angle = pi / (1 << (target - control))
            stage.cp(angle, control, target)
        stages.append((f"target {target}", stage))

    if do_swaps:
        for left in range(num_qubits // 2):
            right = num_qubits - left - 1
            stage = QuantumCircuit(num_qubits, name=f"Swap{left}_{right}")
            stage.swap(left, right)
            stages.append((f"swap {left}<->{right}", stage))

    return stages


# Recursive QFT construction based on the smaller subcircuit.
def build_recursive_qft(num_qubits: int) -> QuantumCircuit:
    """Build the recursive QFT using the recursive construction."""
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative.")

    stages = _recursive_qft_stage_circuits(num_qubits)
    return _compose_stage_circuits(
        num_qubits,
        stages,
        name=f"RecursiveQFT_{num_qubits}",
    )


def _recursive_qft_stage_circuits(num_qubits: int) -> list[tuple[str, QuantumCircuit]]:
    """Build the recursive QFT as labeled stage circuits."""
    return _recursive_qft_stage_circuits_for_qubits(num_qubits, list(range(num_qubits)))


def _recursive_qft_stage_circuits_for_qubits(
    num_qubits: int,
    qubits: list[int],
) -> list[tuple[str, QuantumCircuit]]:
    if not qubits:
        return []

    if len(qubits) == 1:
        stage = QuantumCircuit(num_qubits, name=f"H{qubits[0]}")
        stage.h(qubits[0])
        return [(f"h on {qubits[0]}", stage)]

    stages = _recursive_qft_stage_circuits_for_qubits(num_qubits, qubits[1:])

    target = qubits[0]
    width = len(qubits)
    stage = QuantumCircuit(num_qubits, name=f"RecursiveLayerQ{target}")
    for offset, control in enumerate(qubits[1:]):
        angle = pi / (1 << (width - offset - 1))
        stage.cp(angle, control, target)

    stage.h(target)
    stages.append((f"recursive layer {target}", stage))

    # This adjacent-swap ladder performs the final qubit reordering from the recursive formula.
    for index in range(width - 1):
        left = qubits[index]
        right = qubits[index + 1]
        swap_stage = QuantumCircuit(num_qubits, name=f"RecursiveSwap{left}_{right}")
        swap_stage.swap(left, right)
        stages.append((f"swap {left}<->{right}", swap_stage))

    return stages


# Apply the transform directly to a state vector.
def qft_on_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    """Apply the standard QFT method amplitude map to a state vector."""
    values = np.asarray(amplitudes, dtype=complex)
    size = values.size

    if size == 0 or size & (size - 1):
        raise ValueError("The number of amplitudes must be a non-zero power of 2.")

    norm = np.linalg.norm(values)
    if norm == 0:
        raise ValueError("The amplitude vector must not be the zero vector.")

    normalized = values / norm
    return dft_amplitudes(normalized)
