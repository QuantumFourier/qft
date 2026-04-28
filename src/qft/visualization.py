from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .standard_qft import (
    _compose_stage_circuits,
    _recursive_qft_stage_circuits,
    _standard_qft_stage_circuits,
)


@dataclass(frozen=True)
class QFTStateSnapshot:
    """One intermediate QFT statevector with a human-readable stage label."""

    label: str
    statevector: Statevector


@dataclass(frozen=True)
class QFTDrawResult:
    """Return value for drawing and inspecting a QFT construction."""

    circuit: QuantumCircuit
    drawing: object | str
    intermediate_states: list[QFTStateSnapshot]


def require_pylatexenc_for_drawing() -> None:
    """Ensure Qiskit's Matplotlib circuit drawer has its LaTeX text helper."""
    try:
        import pylatexenc  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Circuit drawing requires pylatexenc. Install it with "
            "`python -m pip install pylatexenc` or reinstall this package with "
            "`python -m pip install -e .` from the repository root."
        ) from exc


def prepare_circuit_for_display(
    circuit: QuantumCircuit,
    decompose_reps: int = 0,
) -> QuantumCircuit:
    """Expand custom blocks before drawing so the circuit structure is easier to read."""
    expanded = circuit

    for _ in range(max(0, decompose_reps)):
        try:
            next_expanded = expanded.decompose()
        except Exception:
            break

        if next_expanded == expanded:
            break

        expanded = next_expanded

    return expanded


def draw_circuit_with_pylatexenc(
    circuit: QuantumCircuit,
    *,
    decompose_reps: int = 0,
    scale: float = 1.5,
    fold: int = -1,
    idle_wires: bool = False,
) -> object:
    """Draw a circuit with Qiskit's Matplotlib drawer backed by pylatexenc."""
    require_pylatexenc_for_drawing()
    shown = prepare_circuit_for_display(circuit, decompose_reps=decompose_reps)
    return shown.draw(
        output="mpl",
        scale=scale,
        fold=fold,
        idle_wires=idle_wires,
    )


def draw_circuit_text(
    circuit: QuantumCircuit,
    *,
    decompose_reps: int = 0,
    fold: int = 160,
    idle_wires: bool = False,
) -> str:
    """Return a terminal-safe circuit drawing."""
    shown = prepare_circuit_for_display(circuit, decompose_reps=decompose_reps)
    return str(shown.draw(output="text", fold=fold, idle_wires=idle_wires))


def show_circuit(
    circuit: QuantumCircuit,
    title: str,
    *,
    display_fn=None,
    decompose_reps: int = 0,
    scale: float = 1.5,
    fold: int = -1,
    idle_wires: bool = False,
) -> None:
    """Display a circuit in notebooks, with a text fallback for plain terminals."""
    print(f"\n{title}")

    try:
        if display_fn is None:
            try:
                from IPython.display import display as display_fn
            except Exception:
                display_fn = None

        figure = draw_circuit_with_pylatexenc(
            circuit,
            decompose_reps=decompose_reps,
            scale=scale,
            fold=fold,
            idle_wires=idle_wires,
        )
        if display_fn is not None:
            display_fn(figure)
        else:
            figure.show()
    except Exception:
        print(
            draw_circuit_text(
                circuit,
                decompose_reps=decompose_reps,
                fold=160,
                idle_wires=idle_wires,
            )
        )


def _default_statevector(num_qubits: int) -> Statevector:
    if num_qubits == 0:
        return Statevector(np.array([1.0], dtype=complex))
    return Statevector.from_label("0" * num_qubits)


def _initial_statevector(num_qubits: int, amplitudes: np.ndarray | None) -> Statevector:
    if amplitudes is None:
        return _default_statevector(num_qubits)

    values = np.asarray(amplitudes, dtype=complex)
    expected_size = 1 << num_qubits
    if values.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} amplitudes for {num_qubits} qubits, got {values.size}."
        )

    norm = np.linalg.norm(values)
    if norm == 0:
        raise ValueError("The amplitude vector must not be the zero vector.")

    return Statevector(values / norm)


def _qft_stages(
    num_qubits: int,
    *,
    do_swap: bool,
    recursive: bool,
) -> list[tuple[str, QuantumCircuit]]:
    if recursive:
        return _recursive_qft_stage_circuits(num_qubits)
    return _standard_qft_stage_circuits(num_qubits, do_swaps=do_swap)


def draw_qft(
    num_qubits: int,
    do_swap: bool = True,
    recursive: bool = False,
    *,
    show_barriers: bool = False,
    show_intermediate_states: bool = False,
    amplitudes: np.ndarray | None = None,
    output: str = "text",
    decompose_reps: int = 0,
    scale: float = 1.5,
    fold: int = -1,
    idle_wires: bool = False,
) -> QFTDrawResult:
    """Draw a QFT circuit and optionally return intermediate statevectors."""
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative.")

    if output not in {"text", "mpl"}:
        raise ValueError("output must be either 'text' or 'mpl'.")

    stages = _qft_stages(
        num_qubits,
        do_swap=do_swap,
        recursive=recursive,
    )
    circuit_name = (
        f"RecursiveQFT_{num_qubits}" if recursive else f"QFT_{num_qubits}"
    )
    circuit = _compose_stage_circuits(
        num_qubits,
        stages,
        name=circuit_name,
        show_barriers=show_barriers,
    )

    if output == "mpl":
        drawing = draw_circuit_with_pylatexenc(
            circuit,
            decompose_reps=decompose_reps,
            scale=scale,
            fold=fold,
            idle_wires=idle_wires,
        )
    else:
        text_fold = 160 if fold < 0 else fold
        drawing = draw_circuit_text(
            circuit,
            decompose_reps=decompose_reps,
            fold=text_fold,
            idle_wires=idle_wires,
        )

    intermediate_states: list[QFTStateSnapshot] = []
    if show_intermediate_states:
        statevector = _initial_statevector(num_qubits, amplitudes)
        for label, stage in stages:
            statevector = statevector.evolve(stage)
            intermediate_states.append(QFTStateSnapshot(label=label, statevector=statevector))

    return QFTDrawResult(
        circuit=circuit,
        drawing=drawing,
        intermediate_states=intermediate_states,
    )
