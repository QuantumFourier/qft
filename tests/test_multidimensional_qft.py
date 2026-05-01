from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
for candidate in (PARENT_DIR, CURRENT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from multidimensional_qft import (
    build_multidimensional_qft,
    expected_multidimensional_qft_state,
    padded_shape,
    prepare_multidimensional_input,
    validate_shape,
)


TEST_SHAPES = ((2, 2), (4, 2), (2, 2, 2), (3, 2), (1, 2), (2, 1))


def assert_multidimensional_qft_matches_expected(shape: tuple[int, ...], method: str) -> None:
    sample_array = np.arange(1, np.prod(shape) + 1, dtype=float).reshape(shape, order="F")
    _, amplitudes = prepare_multidimensional_input(sample_array)
    expected_state = expected_multidimensional_qft_state(sample_array)
    total_qubits = int(np.log2(np.prod(padded_shape(shape))))

    multidimensional_qft = build_multidimensional_qft(shape, method=method)
    circuit = QuantumCircuit(total_qubits)
    circuit.initialize(amplitudes, range(total_qubits))
    circuit.compose(multidimensional_qft, inplace=True)
    actual_state = Statevector.from_instruction(circuit).data
    assert np.allclose(actual_state, expected_state)


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("method", ("standard", "recursive"))
def test_multidimensional_qft_matches_classical_dft(shape: tuple[int, ...], method: str) -> None:
    assert_multidimensional_qft_matches_expected(shape, method)


def test_validate_shape_rejects_empty_or_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError):
        validate_shape(())

    with pytest.raises(ValueError):
        validate_shape((2, 0))

    with pytest.raises(ValueError):
        validate_shape((2, -1))


def test_invalid_multidimensional_qft_method_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_multidimensional_qft((2, 2), method="invalid")


# Verify the multidimensional QFT against the classical multidimensional DFT.
def main() -> None:
    for shape in TEST_SHAPES:
        for method in ("standard", "recursive"):
            assert_multidimensional_qft_matches_expected(shape, method)

    print("All multidimensional QFT checks passed.")


if __name__ == "__main__":
    main()
