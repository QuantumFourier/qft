# QFT-Library

This library contains Python code for comparing the standard QFT method, the recursive QFT, a distributed-QFT cost model, and multidimensional QFT.

## Local Environment Installation

Clone the repository first:

```bash
git clone https://github.com/QuantumFourier/qft.git
cd qft
```

Then install the package in editable mode from the local repository path. From a terminal, use:

```bash
python -m pip install -e /path/to/repository
```

In a notebook, use:

```python
%pip install -e /path/to/repository
```

This makes the `qft` package importable while keeping local changes under `src/qft/` available immediately.

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

The package also exposes a single dispatcher for the main build modes:

```python
standard = qft.build(kind="standard", num_qubits=4, do_swap=True, recursive=False)
multidimensional = qft.build(kind="multidimensional", shape=(4, 2), method="recursive")
distributed = qft.build(kind="distributed", num_qubits=4, num_nodes=2, strategy="contiguous")
```

- `kind="standard"` returns a one-dimensional QFT `QuantumCircuit`.
- `kind="multidimensional"` returns a multidimensional QFT `QuantumCircuit`.
- `kind="distributed"` returns a `DistributedQFTBuildResult` with:
  - the standard and recursive circuits
  - the chosen node mapping
  - distributed-cost reports
  - the recommended method under the built-in cost model

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

## Package Structure

The importable package lives in `src/qft/`.

- `src/qft/__init__.py`
  Public package entry point. It exposes:
  - `qft.qft(...)`
  - `qft.build(...)`
  - the main multidimensional and distributed helper functions

- `src/qft/standard_qft.py`
  Core one-dimensional QFT implementations:
  - iterative circuit construction
  - recursive circuit construction
  - amplitude-level reference transforms

- `src/qft/visualization.py`
  Circuit drawing and QFT stage inspection:
  - text and Matplotlib drawing
  - optional barriers
  - intermediate statevector snapshots

- `src/qft/multidimensional.py`
  Multidimensional QFT utilities:
  - shape validation
  - padding to powers of two
  - multidimensional state preparation
  - multidimensional QFT circuit construction
  - expected classical transform for comparison

- `src/qft/distributed.py`
  Distributed-QFT analysis utilities:
  - node assignment
  - logical-to-physical mapping
  - non-local gate analysis
  - communication-cost comparison between standard and recursive QFT
  - `build_distributed_qft(...)` for the packaged distributed result object

- `src/qft/distributed_blocks.py`
  Visualization-oriented distributed building blocks:
  - Bell-pair resource
  - cat-entangler / cat-disentangler
  - non-local controlled phase
  - teleportation leg
  - teleportation-based swap

- `src/qft/sampler_utils.py`
  Shared Aer and fake-backend sampling helpers used by the benchmarking and distributed analysis tools.

## Tests

The test suite now lives under `tests/`:

- `tests/test_qft_package.py`
- `tests/test_forward_qft.py`
- `tests/test_multidimensional_qft.py`
- `tests/test_distributed_qft.py`

Run the full suite with:

```bash
python -m pytest tests
```

## Project Utilities

- `qft_implementation_metrics.py`
  Records implementation metrics such as build time, transpilation time, gate counts, and estimated backend cost.

- `environment_benchmark.py`
  Runs broader benchmarking sweeps across the standard, multidimensional, and distributed implementations.

- `export_backend_properties.py`
  Exports fake backend calibration data to JSON when needed.

## Main Commands

Run the distributed QFT comparison:

```bash
python -m qft.distributed
```

Run the distributed comparison with custom settings:

```bash
python -m qft.distributed --qubits 6 --nodes 3 --strategy contiguous --shots 128
python -m qft.distributed --qubits 6 --nodes 3 --strategy interleaved --shots 128 --show-full-log
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

- `pylatexenc`, `matplotlib`, `qiskit-aer`, and `qiskit-ibm-runtime` are installed as package dependencies so the package utilities can draw circuits, run Aer simulations, and use Qiskit fake backends.

- If a notebook still reports a missing dependency after selecting a new environment, reinstall this project into that environment:

```bash
python -m pip install -e .
```
