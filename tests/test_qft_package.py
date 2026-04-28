from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import qft as qft_package
from qft import (
    build_recursive_qft,
    build_standard_qft,
    draw_qft,
    qft,
    qft_on_amplitudes,
)
from qft.recursive import qft as recursive_qft_alias
from qft.standard import qft as standard_qft_alias


@pytest.mark.parametrize("num_qubits", range(0, 6))
def test_standard_package_circuit_matches_qiskit_reference(num_qubits: int) -> None:
    package_circuit = build_standard_qft(num_qubits)

    if num_qubits == 0:
        assert package_circuit.num_qubits == 0
        return

    reference = QuantumCircuit(num_qubits)
    reference.append(QFTGate(num_qubits), range(num_qubits))
    assert Operator(package_circuit).equiv(Operator(reference))


@pytest.mark.parametrize("num_qubits", range(0, 6))
def test_recursive_package_circuit_matches_qiskit_reference(num_qubits: int) -> None:
    package_circuit = build_recursive_qft(num_qubits)

    if num_qubits == 0:
        assert package_circuit.num_qubits == 0
        return

    reference = QuantumCircuit(num_qubits)
    reference.append(QFTGate(num_qubits), range(num_qubits))
    assert Operator(package_circuit).equiv(Operator(reference))


def test_qft_alias_returns_same_circuit() -> None:
    assert Operator(qft(4)).equiv(Operator(build_standard_qft(4)))
    assert Operator(standard_qft_alias(4)).equiv(Operator(build_standard_qft(4)))
    assert Operator(recursive_qft_alias(4)).equiv(Operator(build_recursive_qft(4)))
    assert Operator(build_standard_qft(4, recursive=True)).equiv(
        Operator(build_recursive_qft(4))
    )


def test_qft_accepts_professor_style_public_arguments() -> None:
    n = 4
    do_swap = False
    recursive = False

    public_circuit = qft_package.qft(n, do_swap, recursive)
    package_function = qft(n, do_swap, recursive)
    keyword_form = qft(n, do_swap=do_swap, recursive=recursive)

    expected = build_standard_qft(n, do_swaps=do_swap)
    assert Operator(public_circuit).equiv(Operator(expected))
    assert Operator(package_function).equiv(Operator(expected))
    assert Operator(keyword_form).equiv(Operator(expected))


def test_draw_qft_adds_barriers_only_when_requested() -> None:
    plain = draw_qft(3, output="text")
    with_barriers = draw_qft(3, show_barriers=True, output="text")

    assert plain.circuit.count_ops().get("barrier", 0) == 0
    assert with_barriers.circuit.count_ops().get("barrier", 0) == 3
    assert isinstance(with_barriers.drawing, str)


def test_draw_qft_intermediate_states_match_standard_final_state() -> None:
    amplitudes = np.arange(1, 9, dtype=float)
    result = draw_qft(
        3,
        do_swap=False,
        show_intermediate_states=True,
        amplitudes=amplitudes,
        output="text",
    )

    circuit = QuantumCircuit(3)
    circuit.initialize(amplitudes / np.linalg.norm(amplitudes), range(3))
    circuit.compose(build_standard_qft(3, do_swaps=False), inplace=True)
    expected = Statevector.from_instruction(circuit)

    assert len(result.intermediate_states) == 3
    assert np.allclose(result.intermediate_states[-1].statevector.data, expected.data)


def test_draw_qft_intermediate_states_match_recursive_final_state() -> None:
    amplitudes = np.arange(1, 9, dtype=float)
    result = draw_qft(
        3,
        recursive=True,
        show_intermediate_states=True,
        amplitudes=amplitudes,
        output="text",
    )

    circuit = QuantumCircuit(3)
    circuit.initialize(amplitudes / np.linalg.norm(amplitudes), range(3))
    circuit.compose(build_recursive_qft(3), inplace=True)
    expected = Statevector.from_instruction(circuit)

    assert len(result.intermediate_states) > 0
    assert np.allclose(result.intermediate_states[-1].statevector.data, expected.data)


def test_draw_qft_rejects_bad_amplitude_length() -> None:
    with pytest.raises(ValueError):
        draw_qft(3, amplitudes=np.array([1.0, 0.0]), show_intermediate_states=True)


def test_amplitude_transform_matches_circuit_action() -> None:
    amplitudes = np.arange(1, 9, dtype=float)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)

    circuit = QuantumCircuit(3)
    circuit.initialize(amplitudes, range(3))
    circuit.compose(build_standard_qft(3), inplace=True)

    actual = Statevector.from_instruction(circuit).data
    expected = qft_on_amplitudes(amplitudes)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("num_qubits", [-1, -2])
def test_negative_qubits_are_rejected(num_qubits: int) -> None:
    with pytest.raises(ValueError):
        build_standard_qft(num_qubits)

    with pytest.raises(ValueError):
        build_standard_qft(num_qubits, recursive=True)
