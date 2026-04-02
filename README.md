# QFT-Implementation-Paper

This folder contains Python code for comparing the standard QFT method, the recursive QFT, a distributed-QFT cost model, and multidimensional QFT.

## Folder Layout

- `standard_recursive_qft/`
  Contains the standard QFT method and recursive QFT implementation, demo, notebook, and test.

- `distributed_qft/`
  Contains the distributed QFT comparison script, notebook, and generated JSON reports.

- `multidimensional_qft/`
  Contains the multidimensional QFT implementation, demo, notebook, and test.

- Root `Implementation/`
  Keeps shared utilities, metrics scripts, backend export tools, and shared JSON outputs.

## Files

- `standard_recursive_qft/forward_qft.py`
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

- `standard_recursive_qft/forward_qft_transpile_demo.py`
  Main demo script for the standard QFT method and recursive QFT. It:
  - builds both circuits
  - compares them to Qiskit's built-in QFT
  - shows amplitudes
  - runs Aer simulation
  - transpiles the circuits for a fake backend

- `standard_recursive_qft/standard_recursive_qft_notebook.ipynb`
  Notebook version of the standard and recursive QFT demo with step-by-step outputs and visualizations.

- `standard_recursive_qft/test_forward_qft.py`
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

- `distributed_qft/distributed_qft_notebook.ipynb`
  Notebook version of the distributed QFT comparison with plots, counts, and execution-log output.

- `qft_implementation_metrics.py`
  Generates implementation metrics such as:
  - build time
  - transpilation time
  - gate counts
  - CNOT counts
  - estimated backend cost

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
python standard_recursive_qft/test_forward_qft.py
```

Run the main QFT demo:

```bash
python standard_recursive_qft/forward_qft_transpile_demo.py
```

Run the main QFT demo with custom settings:

```bash
python standard_recursive_qft/forward_qft_transpile_demo.py --qubits 3 --shots 64 --method both
python standard_recursive_qft/forward_qft_transpile_demo.py --qubits 4 --shots 128 --method standard
python standard_recursive_qft/forward_qft_transpile_demo.py --qubits 4 --shots 128 --method recursive
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

Export backend properties:

```bash
python export_backend_properties.py
```

## Notes

- If `qiskit-aer` is missing, it can be installed using the following command:

```bash
python -m pip install qiskit-aer
```
