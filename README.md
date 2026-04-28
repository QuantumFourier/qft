# QFT-Library

This library contains Python code for comparing the standard QFT method, the recursive QFT, a distributed-QFT cost model, and multidimensional QFT.

## Local Environment Installation

Clone the GitHub repository first:

```bash
git clone https://github.com/QuantumFourier/qft.git
cd qft
```

Install the package in editable mode from the repository root so local changes under `src/qft/` are available immediately. In a notebook opened from the cloned repository, use:

```python
%pip install -e .
```

From a terminal opened at the repository root, the equivalent command is:

```bash
python -m pip install -e .
```

## Importing the QFT Library

```python
import qft
```

## How to Use QFT

The package exposes a convenience function that builds a Qiskit `QuantumCircuit`:

```python
circuit = qft.qft(n, do_swap=True, recursive=False)
```

- `n` is the number of qubits.
- `do_swap` controls whether the standard QFT circuit performs the final qubit-reversal swaps.
- `recursive=True` builds the recursive QFT implementation through the same public entry point. In this mode, `do_swap` is ignored because the recursive construction performs its own final qubit reordering.

Examples:

```python
standard_circuit = qft.qft(4)
standard_without_swaps = qft.qft(4, do_swap=False)
recursive_circuit = qft.qft(4, recursive=True)
```

## Drawing and Intermediate States

The package also provides a drawing helper that can insert barriers between logical stages and return intermediate statevectors after each stage:

```python
result = qft.draw_qft(
    4,
    do_swap=True,
    recursive=False,
    show_barriers=True,
    show_intermediate_states=True,
    output="text",
)
```

- `result.circuit` is the constructed QFT circuit, including barriers if requested.
- `result.drawing` is the rendered circuit output as text or a Matplotlib figure.
- `result.intermediate_states` is a list of labeled statevector snapshots, one per stage of the QFT construction.

Example:

```python
amplitudes = [1, 2, 3, 4]
result = qft.draw_qft(
    2,
    show_barriers=True,
    show_intermediate_states=True,
    amplitudes=amplitudes,
)

print(result.drawing)
for snapshot in result.intermediate_states:
    print(snapshot.label)
    print(snapshot.statevector)
```

## Folder Layout

- `src/qft/`
  Contains the importable Python package. The public API is exposed from `qft.__init__`, and the standard and recursive QFT builders live in `src/qft/standard_qft.py`.

- `standard_qft/`
  Contains the standard and recursive QFT demo, notebook, and test scripts.

- `distributed_qft/`
  Contains the distributed QFT comparison script, notebook, illustrative non-local subcircuits, and generated JSON reports.

- `multidimensional_qft/`
  Contains the multidimensional QFT implementation, demo, notebook, and test.

- Root `Implementation/`
  Keeps shared utilities, metrics scripts, backend export tools, and shared JSON outputs.

## Files

- `src/qft/standard_qft.py`
  Contains the core QFT implementations:
  - standard QFT method
  - recursive QFT
  - direct DFT-based amplitude check

- `multidimensional_qft/multidimensional_qft.py`
  Contains the multidimensional QFT implementation. It:
  - validates multidimensional shapes
  - pads non-power-of-two dimensions with zeros up to the next power of two
  - flattens arrays into quantum-state vectors
  - applies one QFT block per dimension
  - computes the expected multidimensional DFT result for comparison

- `standard_qft/forward_qft_transpile_demo.py`
  Main demo script for the standard QFT method and recursive QFT. It:
  - builds both circuits
  - compares them to Qiskit's built-in QFT
  - shows amplitudes
  - runs Aer simulation
  - transpiles the circuits for a fake backend

- `standard_qft/standard_qft_notebook.ipynb`
  Notebook version of the standard and recursive QFT demo with step-by-step outputs and visualizations.

- `standard_qft/test_forward_qft.py`
  Correctness test script. It checks that:
  - the QFT circuits match Qiskit's reference implementation
  - the output amplitudes match the expected transform

- `multidimensional_qft/multidimensional_qft_demo.py`
  Main multidimensional QFT demo. It:
  - builds a multidimensional input array
  - encodes it into a quantum state
  - runs the standard and recursive multidimensional QFT circuits
  - compares them with the expected multidimensional DFT
  - runs ideal and noisy Aer simulations
  - shows transpilation results

- `multidimensional_qft/multidimensional_qft_notebook.ipynb`
  Notebook version of the multidimensional QFT demo, including padded examples and visualizations.

- `multidimensional_qft/test_multidimensional_qft.py`
  Correctness test for the multidimensional QFT implementation.

- `distributed_qft/distributed_qft_comparison.py`
  Compares the standard QFT method and recursive QFT in a distributed setting. It:
  - splits qubits across nodes
  - identifies local and non-local gates
  - estimates communication cost
  - runs Aer simulations
  - prints a recommendation

- `distributed_qft/distributed_qft_blocks.py`
  Contains explicit visualization-oriented distributed building blocks, including:
  - cat-entangler
  - cat-disentangler
  - entanglement-assisted non-local controlled phase
  - teleportation leg
  - teleportation-based distributed swap

- `distributed_qft/distributed_qft_notebook.ipynb`
  Notebook version of the distributed QFT comparison with plots, counts, and execution-log output.

- `qft_implementation_metrics.py`
  Generates implementation metrics such as:
  - build time
  - transpilation time
  - gate counts
  - CNOT counts
  - estimated backend cost

- `environment_benchmark.py`
  Saves cross-environment benchmarking reports for laptop and HPC comparison. It records:
  - runtime
  - peak memory use
  - maximum problem size handled from the tested sweep
  - stability and reproducibility across repeated seeded runs
  - ideal-Aer and noisy-Aer measurement summaries

- `qft_sampler_utils.py`
  Shared helper functions for:
  - building sample input states
  - measured circuits
  - Aer execution
  - counts summaries

- `export_backend_properties.py`
  Exports fake backend calibration data to JSON.

- `fake_manila_v2_backend_properties.json`
  Exported backend properties for the fake Manila backend.

- `distributed_qft/distributed_qft_comparison.json`
  Example output report produced by `distributed_qft_comparison.py`.

- `distributed_qft/distributed_qft_comparison_6q_3nodes.json`
  Example distributed comparison report for a 6-qubit, 3-node case.

- `qft_implementation_metrics.json`
  Example metrics output produced by `qft_implementation_metrics.py`.

## Main Commands

Run the QFT correctness test:

```bash
python standard_qft/test_forward_qft.py
```

Run the main QFT demo:

```bash
python standard_qft/forward_qft_transpile_demo.py
```

Run the main QFT demo with custom settings:

```bash
python standard_qft/forward_qft_transpile_demo.py --qubits 3 --shots 64 --method both
python standard_qft/forward_qft_transpile_demo.py --qubits 4 --shots 128 --method standard
python standard_qft/forward_qft_transpile_demo.py --qubits 4 --shots 128 --method recursive
```

Run the distributed QFT comparison:

```bash
python distributed_qft/distributed_qft_comparison.py
```

Run the distributed comparison with custom settings:

```bash
python distributed_qft/distributed_qft_comparison.py --qubits 6 --nodes 3 --strategy contiguous --shots 128
python distributed_qft/distributed_qft_comparison.py --qubits 6 --nodes 3 --strategy interleaved --shots 128 --show-full-log
```

Run the multidimensional QFT demo:

```bash
python multidimensional_qft/multidimensional_qft_demo.py
python multidimensional_qft/multidimensional_qft_demo.py --shape 4 2 --shots 128 --method both
python multidimensional_qft/multidimensional_qft_demo.py --shape 4 2 2 --shots 128 --method both
python multidimensional_qft/multidimensional_qft_demo.py --shape 3 2 --shots 64 --method both
python multidimensional_qft/multidimensional_qft_demo.py --shape 9 4 --shots 128 --method both
python multidimensional_qft/multidimensional_qft_demo.py --shape 8 8 --shots 256 --method standard
```

Run the multidimensional QFT test:

```bash
python multidimensional_qft/test_multidimensional_qft.py
```

Generate the metrics report:

```bash
python qft_implementation_metrics.py
python qft_implementation_metrics.py --qubits 4 --optimization-level 3
```

Generate a laptop/HPC comparison report:

```bash
python environment_benchmark.py --label laptop --output laptop_benchmark.json
python environment_benchmark.py --label hpc --output hpc_benchmark.json
python environment_benchmark.py --experiments standard_qft --standard-qubits 3 4 5 6 --repeats 5 --shots 512 --label hpc --output hpc_standard_benchmark.json
python environment_benchmark.py --experiments multidimensional_qft --multidimensional-shapes 4x2 4x2x2 8x8 --repeats 5 --shots 512 --label laptop --output laptop_multidim_benchmark.json
```

Export backend properties:

```bash
python export_backend_properties.py
```

## Notes

- `pylatexenc`, `matplotlib`, `qiskit-aer`, and `qiskit-ibm-runtime` are installed as package dependencies so the demos and notebooks can draw circuits, run Aer simulations, and use Qiskit fake backends.

- If a notebook still reports a missing dependency after selecting a new environment, reinstall this project into that environment:

```bash
python -m pip install -e .
```
