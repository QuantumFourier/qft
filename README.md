# QFT-Implementation-Paper

This folder contains Python code for comparing the standard QFT method, the recursive QFT, and a distributed-QFT cost model.

## Files

- `forward_qft.py`
  Contains the core QFT implementations:
  - standard QFT method
  - recursive QFT
  - direct DFT-based amplitude check

- `forward_qft_transpile_demo.py`
  Main demo script for the standard QFT method and recursive QFT. It:
  - builds both circuits
  - compares them to Qiskit's built-in QFT
  - shows amplitudes
  - runs Aer simulation
  - transpiles the circuits for a fake backend

- `test_forward_qft.py`
  Correctness test script. It checks that:
  - the QFT circuits match Qiskit's reference implementation
  - the output amplitudes match the expected transform

- `distributed_qft_comparison.py`
  Compares the standard QFT method and recursive QFT in a distributed setting. It:
  - splits qubits across nodes
  - identifies local and non-local gates
  - estimates communication cost
  - runs Aer simulations
  - prints a recommendation

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

- `distributed_qft_comparison.json`
  Example output report produced by `distributed_qft_comparison.py`.

- `distributed_qft_comparison_6q_3nodes.json`
  Example distributed comparison report for a 6-qubit, 3-node case.

- `qft_implementation_metrics.json`
  Example metrics output produced by `qft_implementation_metrics.py`.

## Main Commands

Run the QFT correctness test:

```bash
python test_forward_qft.py
```

Run the main QFT demo:

```bash
python forward_qft_transpile_demo.py
```

Run the main QFT demo with custom settings:

```bash
python forward_qft_transpile_demo.py --qubits 3 --shots 64 --method both
python forward_qft_transpile_demo.py --qubits 4 --shots 128 --method standard
python forward_qft_transpile_demo.py --qubits 4 --shots 128 --method recursive
```

Run the distributed QFT comparison:

```bash
python distributed_qft_comparison.py
```

Run the distributed comparison with custom settings:

```bash
python distributed_qft_comparison.py --qubits 6 --nodes 3 --strategy contiguous --shots 128
python distributed_qft_comparison.py --qubits 6 --nodes 3 --strategy interleaved --shots 128 --show-full-log
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
