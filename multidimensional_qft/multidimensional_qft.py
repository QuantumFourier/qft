from __future__ import annotations

from math import log2
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
SRC_DIR = PARENT_DIR / "src"
for candidate in (PARENT_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from qft.standard_recursive_qft import build_recursive_qft, build_standard_qft


# Round a positive integer up to the next power of two.
def next_power_of_two(size: int) -> int:
    if size < 1:
        raise ValueError("Each dimension size must be a positive integer.")
    if size & (size - 1) == 0:
        return size
    return 1 << (size - 1).bit_length()


# Check that every dimension size is a positive integer.
def validate_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        raise ValueError("The multidimensional shape must contain at least one dimension.")

    normalized = tuple(int(size) for size in shape)
    for size in normalized:
        if size < 1:
            raise ValueError("Each dimension size must be a positive integer.")

    return normalized


# Pad each dimension size up to the next power of two.
def padded_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    checked_shape = validate_shape(shape)
    return tuple(next_power_of_two(size) for size in checked_shape)


# Convert each dimension size into the number of qubits it needs.
def dimension_qubit_widths(shape: tuple[int, ...]) -> list[int]:
    checked_shape = padded_shape(shape)
    return [int(log2(size)) for size in checked_shape]


# Flatten the array so the first dimension changes fastest.
def flatten_array(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=complex).reshape(-1, order="F")


# Pad an array with zeros so each dimension reaches a chosen target shape.
def pad_array(array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    data = np.asarray(array, dtype=complex)
    if data.ndim != len(target_shape):
        raise ValueError("The target shape must have the same number of dimensions as the array.")

    if any(current > target for current, target in zip(data.shape, target_shape)):
        raise ValueError("The target shape must be at least as large as the current shape in every dimension.")

    padded = np.zeros(target_shape, dtype=complex)
    slices = tuple(slice(0, size) for size in data.shape)
    padded[slices] = data
    return padded


# Pad an array so every dimension reaches the next power of two.
def pad_array_to_powers_of_two(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=complex)
    return pad_array(data, padded_shape(data.shape))


# Normalize the flattened array and treat it as a quantum-state vector.
def encode_array_as_state(array: np.ndarray, *, pad_to_power_of_two: bool = True) -> np.ndarray:
    data = pad_array_to_powers_of_two(array) if pad_to_power_of_two else np.asarray(array, dtype=complex)
    flattened = flatten_array(data)
    norm = np.linalg.norm(flattened)
    if norm == 0:
        raise ValueError("The input array must not be the zero array.")
    return flattened / norm


# Build the padded array and the corresponding normalized amplitude vector.
def prepare_multidimensional_input(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = pad_array_to_powers_of_two(array)
    return padded, encode_array_as_state(padded, pad_to_power_of_two=False)


# Apply the multidimensional DFT one axis at a time.
def multidimensional_dft(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=complex)
    transformed = data.copy()
    for axis in range(transformed.ndim):
        transformed = np.fft.ifft(transformed, axis=axis, norm="ortho")
    return transformed


# Build the full multidimensional QFT circuit from one-dimensional QFT blocks.
def build_multidimensional_qft(shape: tuple[int, ...], method: str = "standard") -> QuantumCircuit:
    checked_shape = padded_shape(shape)
    widths = dimension_qubit_widths(checked_shape)
    registers = [QuantumRegister(width, f"d{index}") for index, width in enumerate(widths)]
    circuit = QuantumCircuit(*registers, name=f"MDQFT_{method}")

    if method == "standard":
        builder = build_standard_qft
    elif method == "recursive":
        builder = build_recursive_qft
    else:
        raise ValueError("method must be 'standard' or 'recursive'.")

    start = 0
    for width in widths:
        block = builder(width)
        circuit.compose(block, qubits=circuit.qubits[start : start + width], inplace=True)
        start += width

    return circuit


# Apply the multidimensional DFT and return the expected state amplitudes.
def expected_multidimensional_qft_state(array: np.ndarray) -> np.ndarray:
    padded = pad_array_to_powers_of_two(array)
    transformed = multidimensional_dft(padded)
    return encode_array_as_state(transformed, pad_to_power_of_two=False)
