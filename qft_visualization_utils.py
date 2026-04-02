from __future__ import annotations

from qiskit import QuantumCircuit


# Expand custom blocks before drawing so the circuit structure is easier to read.
def prepare_circuit_for_display(circuit: QuantumCircuit, decompose_reps: int = 0) -> QuantumCircuit:
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


# Draw a clearer circuit diagram for notebooks, with a text fallback for plain terminals.
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
    print(f"\n{title}")

    shown = prepare_circuit_for_display(circuit, decompose_reps=decompose_reps)

    try:
        if display_fn is None:
            try:
                from IPython.display import display as display_fn
            except Exception:
                display_fn = None

        figure = shown.draw(
            "mpl",
            scale=scale,
            fold=fold,
            idle_wires=idle_wires,
        )
        if display_fn is not None:
            display_fn(figure)
        else:
            figure.show()
    except Exception:
        print(shown.draw(output="text", fold=160, idle_wires=idle_wires))
