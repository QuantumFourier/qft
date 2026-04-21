from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import traceback
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeFez

CURRENT_DIR = Path(__file__).resolve().parent
STANDARD_DIR = CURRENT_DIR / "standard_recursive_qft"
MULTIDIMENSIONAL_DIR = CURRENT_DIR / "multidimensional_qft"
DISTRIBUTED_DIR = CURRENT_DIR / "distributed_qft"
for candidate in (CURRENT_DIR, STANDARD_DIR, MULTIDIMENSIONAL_DIR, DISTRIBUTED_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from distributed_qft_comparison import analyze_distributed_costs, build_node_mapping, choose_best_method
from forward_qft import build_recursive_qft, build_standard_qft, qft_on_amplitudes
from multidimensional_qft import (
    build_multidimensional_qft,
    dimension_qubit_widths,
    expected_multidimensional_qft_state,
    padded_shape,
    prepare_multidimensional_input,
)
from qft_sampler_utils import (
    build_measured_qft_circuit,
    build_sample_amplitudes,
    counts_summary,
    sample_aer_counts,
    sample_noisy_aer_counts,
    select_fake_backend,
    top_outcomes,
    total_variation_distance,
)


def parse_shape_token(token: str) -> tuple[int, ...]:
    cleaned = token.lower().replace(",", "x")
    try:
        shape = tuple(int(part) for part in cleaned.split("x") if part)
    except ValueError as exc:
        raise ValueError(f"Invalid shape token: {token}") from exc

    if not shape or any(size < 1 for size in shape):
        raise ValueError(f"Invalid shape token: {token}")
    return shape


def build_seed_list(base_seed: int, repeats: int) -> list[int]:
    return [base_seed + index for index in range(repeats)]


def import_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def current_peak_rss_megabytes() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def problem_size_key(experiment: str, problem_size: dict[str, Any]) -> tuple[int, ...]:
    if experiment in {"standard_qft", "distributed_qft"}:
        return (int(problem_size["num_qubits"]),)

    shape = tuple(problem_size["shape"])
    total_qubits = sum(dimension_qubit_widths(shape))
    return (total_qubits, int(np.prod(shape)), len(shape))


def build_reproducibility_summary(
    counts_runs: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> dict[str, Any]:
    counts_only = [dict(item["counts"]) for item in counts_runs]
    top_bitstrings = [
        top_outcomes(counts, limit=1)[0][0] if counts else None
        for counts in counts_only
    ]
    pairwise_tvd_values = [
        total_variation_distance(left, right)
        for left, right in combinations(counts_only, 2)
    ]

    return {
        "repeats": len(counts_runs),
        "top_outcome_consistent": len(set(top_bitstrings)) <= 1,
        "top_outcomes_by_run": top_bitstrings,
        "pairwise_tvd_values": pairwise_tvd_values,
        "pairwise_tvd_mean": float(np.mean(pairwise_tvd_values)) if pairwise_tvd_values else 0.0,
        "pairwise_tvd_max": max(pairwise_tvd_values, default=0.0),
        "reference_run": counts_summary(counts_only[0], limit=limit) if counts_only else None,
        "runs": counts_runs,
    }


def environment_metadata(label: str | None = None) -> dict[str, Any]:
    return {
        "label": label,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "numpy_version": import_version("numpy"),
        "qiskit_version": import_version("qiskit"),
        "qiskit_aer_version": import_version("qiskit_aer"),
        "qiskit_ibm_runtime_version": import_version("qiskit_ibm_runtime"),
    }


def summarize_case_runs(experiment: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [case for case in cases if case["success"]]
    maximum_problem_size_handled = None
    if successful:
        largest = max(successful, key=lambda item: problem_size_key(experiment, item["problem_size"]))
        maximum_problem_size_handled = largest["problem_size"]

    return {
        "cases": cases,
        "successful_case_count": len(successful),
        "failed_case_count": len(cases) - len(successful),
        "maximum_problem_size_handled": maximum_problem_size_handled,
    }


def sampled_counts_runs(
    measured_circuit: QuantumCircuit,
    *,
    seeds: list[int],
    shots: int,
    backend=None,
) -> list[dict[str, Any]]:
    runs = []
    for seed in seeds:
        if backend is None:
            counts = sample_aer_counts(
                measured_circuit,
                shots=shots,
                seed_simulator=seed,
                seed_transpiler=seed,
            )
        else:
            counts = sample_noisy_aer_counts(
                measured_circuit,
                backend=backend,
                shots=shots,
                seed_simulator=seed,
                seed_transpiler=seed,
            )
        runs.append({"seed": seed, "counts": counts})
    return runs


def benchmark_standard_case(problem_size: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    num_qubits = int(problem_size["num_qubits"])
    amplitudes = build_sample_amplitudes(num_qubits)
    expected_state = qft_on_amplitudes(amplitudes)
    backend = select_fake_backend(num_qubits)
    pass_manager = generate_preset_pass_manager(
        optimization_level=config["optimization_level"],
        backend=backend,
    )
    seeds = build_seed_list(config["base_seed"], config["repeats"])
    methods = []

    for label, builder in (
        ("standard", build_standard_qft),
        ("recursive", build_recursive_qft),
    ):
        build_start = perf_counter()
        qft_circuit = builder(num_qubits)
        build_time = perf_counter() - build_start

        transpile_start = perf_counter()
        transpiled = pass_manager.run(qft_circuit)
        transpile_time = perf_counter() - transpile_start

        state_circuit = QuantumCircuit(num_qubits)
        state_circuit.initialize(amplitudes, range(num_qubits))
        state_circuit.compose(qft_circuit, inplace=True)
        actual_state = Statevector.from_instruction(state_circuit).data

        measured_circuit = build_measured_qft_circuit(amplitudes, qft_circuit)
        ideal_runs = sampled_counts_runs(measured_circuit, seeds=seeds, shots=config["shots"])
        noisy_runs = None
        if config["include_noisy_aer"]:
            noisy_runs = sampled_counts_runs(
                measured_circuit,
                seeds=seeds,
                shots=config["shots"],
                backend=backend,
            )

        methods.append(
            {
                "method": label,
                "build_time_seconds": build_time,
                "transpile_time_seconds": transpile_time,
                "transpiled_depth": transpiled.depth(),
                "transpiled_gate_count": transpiled.size(),
                "statevector_max_abs_error": float(np.max(np.abs(actual_state - expected_state))),
                "ideal_reproducibility": build_reproducibility_summary(ideal_runs),
                "noisy_reproducibility": build_reproducibility_summary(noisy_runs) if noisy_runs is not None else None,
            }
        )

    return {
        "problem_size": {"num_qubits": num_qubits},
        "backend_name": backend.name,
        "methods": methods,
    }


def benchmark_multidimensional_case(problem_size: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    shape = tuple(int(size) for size in problem_size["shape"])
    sample_array = np.arange(1, np.prod(shape) + 1, dtype=float).reshape(shape, order="F")
    padded_array, amplitudes = prepare_multidimensional_input(sample_array)
    expected_state = expected_multidimensional_qft_state(sample_array)
    total_qubits = sum(dimension_qubit_widths(shape))
    backend = select_fake_backend(total_qubits)
    pass_manager = generate_preset_pass_manager(
        optimization_level=config["optimization_level"],
        backend=backend,
    )
    seeds = build_seed_list(config["base_seed"], config["repeats"])
    methods = []

    for label in ("standard", "recursive"):
        build_start = perf_counter()
        qft_circuit = build_multidimensional_qft(shape, method=label)
        build_time = perf_counter() - build_start

        transpile_start = perf_counter()
        transpiled = pass_manager.run(qft_circuit)
        transpile_time = perf_counter() - transpile_start

        state_circuit = QuantumCircuit(total_qubits)
        state_circuit.initialize(amplitudes, range(total_qubits))
        state_circuit.compose(qft_circuit, inplace=True)
        actual_state = Statevector.from_instruction(state_circuit).data

        measured_circuit = build_measured_qft_circuit(amplitudes, qft_circuit)
        ideal_runs = sampled_counts_runs(measured_circuit, seeds=seeds, shots=config["shots"])
        noisy_runs = None
        if config["include_noisy_aer"]:
            noisy_runs = sampled_counts_runs(
                measured_circuit,
                seeds=seeds,
                shots=config["shots"],
                backend=backend,
            )

        methods.append(
            {
                "method": label,
                "build_time_seconds": build_time,
                "transpile_time_seconds": transpile_time,
                "transpiled_depth": transpiled.depth(),
                "transpiled_gate_count": transpiled.size(),
                "statevector_max_abs_error": float(np.max(np.abs(actual_state - expected_state))),
                "ideal_reproducibility": build_reproducibility_summary(ideal_runs),
                "noisy_reproducibility": build_reproducibility_summary(noisy_runs) if noisy_runs is not None else None,
            }
        )

    return {
        "problem_size": {
            "shape": list(shape),
            "padded_shape": list(padded_shape(shape)),
            "total_qubits": total_qubits,
            "array_entries": int(np.prod(shape)),
            "padded_entries": int(np.prod(padded_array.shape)),
        },
        "backend_name": backend.name,
        "methods": methods,
    }


def benchmark_distributed_case(problem_size: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    num_qubits = int(problem_size["num_qubits"])
    num_nodes = int(problem_size.get("num_nodes", config["distributed_nodes"]))
    strategy = str(problem_size.get("strategy", config["distributed_strategy"]))
    node_mapping = build_node_mapping(num_qubits, num_nodes, strategy)
    backend = FakeFez()
    amplitudes = build_sample_amplitudes(num_qubits)
    seeds = build_seed_list(config["base_seed"], config["repeats"])
    methods = []

    for label, builder in (
        ("Standard QFT method", build_standard_qft),
        ("Recursive QFT", build_recursive_qft),
    ):
        build_start = perf_counter()
        qft_circuit = builder(num_qubits)
        build_time = perf_counter() - build_start

        costs = analyze_distributed_costs(qft_circuit, node_mapping)
        measured_circuit = build_measured_qft_circuit(amplitudes, qft_circuit)
        ideal_runs = sampled_counts_runs(measured_circuit, seeds=seeds, shots=config["shots"])
        noisy_runs = None
        if config["include_noisy_aer"]:
            noisy_runs = sampled_counts_runs(
                measured_circuit,
                seeds=seeds,
                shots=config["shots"],
                backend=backend,
            )

        methods.append(
            {
                "method": label,
                "build_time_seconds": build_time,
                "ideal_reproducibility": build_reproducibility_summary(ideal_runs),
                "noisy_reproducibility": build_reproducibility_summary(noisy_runs) if noisy_runs is not None else None,
                **costs,
            }
        )

    best_method = choose_best_method(methods)
    return {
        "problem_size": {
            "num_qubits": num_qubits,
            "num_nodes": num_nodes,
            "strategy": strategy,
        },
        "backend_name": backend.name,
        "node_mapping": node_mapping,
        "recommended_method": best_method["method"],
        "methods": methods,
    }


def benchmark_case(experiment: str, problem_size: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    start = perf_counter()
    try:
        if experiment == "standard_qft":
            payload = benchmark_standard_case(problem_size, config)
        elif experiment == "multidimensional_qft":
            payload = benchmark_multidimensional_case(problem_size, config)
        elif experiment == "distributed_qft":
            payload = benchmark_distributed_case(problem_size, config)
        else:
            raise ValueError(f"Unsupported experiment: {experiment}")

        return {
            "success": True,
            "runtime_seconds": perf_counter() - start,
            "peak_rss_megabytes": current_peak_rss_megabytes(),
            **payload,
        }
    except Exception as exc:
        return {
            "success": False,
            "problem_size": problem_size,
            "runtime_seconds": perf_counter() - start,
            "peak_rss_megabytes": current_peak_rss_megabytes(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }


def worker_command_payload(experiment: str, problem_size: dict[str, Any], config: dict[str, Any]) -> str:
    return json.dumps(
        {
            "experiment": experiment,
            "problem_size": problem_size,
            "config": config,
        }
    )


def run_case_in_subprocess(experiment: str, problem_size: dict[str, Any], config: dict[str, Any], timeout: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-json",
        worker_command_payload(experiment, problem_size, config),
    ]
    environment = dict(os.environ)
    environment.setdefault("MPLBACKEND", "Agg")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "problem_size": problem_size,
            "error_type": "TimeoutExpired",
            "error_message": f"Benchmark case exceeded timeout of {timeout} seconds.",
        }

    if result.returncode != 0:
        return {
            "success": False,
            "problem_size": problem_size,
            "error_type": "WorkerProcessError",
            "error_message": result.stderr.strip() or result.stdout.strip() or "Worker process failed.",
        }

    return json.loads(result.stdout)


def benchmark_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "shots": args.shots,
        "repeats": args.repeats,
        "base_seed": args.base_seed,
        "optimization_level": args.optimization_level,
        "include_noisy_aer": not args.skip_noisy_aer,
        "distributed_nodes": args.distributed_nodes,
        "distributed_strategy": args.distributed_strategy,
    }


def format_problem_size(problem_size: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in problem_size.items())


def run_case_batch(
    experiment: str,
    problem_sizes: list[dict[str, Any]],
    config: dict[str, Any],
    timeout: int,
) -> list[dict[str, Any]]:
    total = len(problem_sizes)
    cases = []

    print(f"\nStarting {experiment} benchmark with {total} case(s)...", flush=True)
    for index, problem_size in enumerate(problem_sizes, start=1):
        print(
            f"[{experiment} {index}/{total}] Running {format_problem_size(problem_size)}",
            flush=True,
        )
        case = run_case_in_subprocess(experiment, problem_size, config, timeout)
        status = "ok" if case["success"] else "failed"
        print(
            f"[{experiment} {index}/{total}] Finished {status} "
            f"in {case.get('runtime_seconds', 0.0):.3f}s "
            f"(peak RSS {case.get('peak_rss_megabytes', 0.0):.1f} MB)",
            flush=True,
        )
        cases.append(case)

    return cases


def print_experiment_summary(experiment: str, summary: dict[str, Any]) -> None:
    print(f"\n{experiment}:")
    print(f"  Successful cases: {summary['successful_case_count']}")
    print(f"  Failed cases: {summary['failed_case_count']}")
    print(f"  Maximum problem size handled: {summary['maximum_problem_size_handled']}")

    for case in summary["cases"]:
        status = "ok" if case["success"] else "failed"
        print(
            f"  - {case['problem_size']} [{status}] "
            f"runtime={case.get('runtime_seconds', 0.0):.3f}s "
            f"peak_rss={case.get('peak_rss_megabytes', 0.0):.1f} MB"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark QFT implementations for cross-environment comparison on laptop and HPC.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("standard_qft", "multidimensional_qft", "distributed_qft", "all"),
        default=["all"],
        help="Which benchmark groups to run.",
    )
    parser.add_argument("--standard-qubits", type=int, nargs="+", default=[3, 4, 5], help="Standard-QFT qubit sizes to benchmark.")
    parser.add_argument(
        "--multidimensional-shapes",
        nargs="+",
        default=["4x2", "4x2x2", "8x8"],
        help="Multidimensional array shapes to benchmark, for example 4x2 or 4x2x2.",
    )
    parser.add_argument("--distributed-qubits", type=int, nargs="+", default=[4, 6], help="Distributed-QFT qubit sizes to benchmark.")
    parser.add_argument("--distributed-nodes", type=int, default=2, help="Default number of distributed nodes.")
    parser.add_argument(
        "--distributed-strategy",
        choices=("contiguous", "interleaved"),
        default="contiguous",
        help="Default qubit placement strategy for distributed benchmarks.",
    )
    parser.add_argument("--shots", type=int, default=256, help="Number of measurement shots per benchmark repeat.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated sampled runs used for stability metrics.")
    parser.add_argument("--base-seed", type=int, default=1234, help="Base seed used to generate repeat seeds.")
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=2,
        help="Qiskit transpilation optimization level.",
    )
    parser.add_argument("--skip-noisy-aer", action="store_true", help="Skip noisy-Aer repeats to speed up the benchmark.")
    parser.add_argument("--label", help="Optional environment label such as laptop or hpc.")
    parser.add_argument(
        "--output",
        default="environment_benchmark_report.json",
        help="Output JSON file for the benchmark report.",
    )
    parser.add_argument("--timeout", type=int, default=900, help="Per-case timeout in seconds.")
    parser.add_argument("--worker-json", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.worker_json:
        payload = json.loads(args.worker_json)
        result = benchmark_case(
            payload["experiment"],
            payload["problem_size"],
            payload["config"],
        )
        print(json.dumps(result))
        return

    selected_experiments = {"standard_qft", "multidimensional_qft", "distributed_qft"} if "all" in args.experiments else set(args.experiments)
    config = benchmark_config_from_args(args)
    report = {
        "environment": environment_metadata(args.label),
        "benchmark_config": {
            **config,
            "standard_qubits": args.standard_qubits,
            "multidimensional_shapes": [list(parse_shape_token(shape)) for shape in args.multidimensional_shapes],
            "distributed_qubits": args.distributed_qubits,
        },
    }

    if "standard_qft" in selected_experiments:
        standard_cases = run_case_batch(
            "standard_qft",
            [{"num_qubits": num_qubits} for num_qubits in args.standard_qubits],
            config,
            args.timeout,
        )
        report["standard_qft"] = summarize_case_runs("standard_qft", standard_cases)

    if "multidimensional_qft" in selected_experiments:
        multidimensional_cases = run_case_batch(
            "multidimensional_qft",
            [{"shape": list(parse_shape_token(shape))} for shape in args.multidimensional_shapes],
            config,
            args.timeout,
        )
        report["multidimensional_qft"] = summarize_case_runs("multidimensional_qft", multidimensional_cases)

    if "distributed_qft" in selected_experiments:
        distributed_cases = run_case_batch(
            "distributed_qft",
            [
                {
                    "num_qubits": num_qubits,
                    "num_nodes": args.distributed_nodes,
                    "strategy": args.distributed_strategy,
                }
                for num_qubits in args.distributed_qubits
            ],
            config,
            args.timeout,
        )
        report["distributed_qft"] = summarize_case_runs("distributed_qft", distributed_cases)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = CURRENT_DIR / output_path

    output_path.write_text(json.dumps(report, indent=2))

    print(f"Saved environment benchmark report to {output_path}")
    for experiment in ("standard_qft", "multidimensional_qft", "distributed_qft"):
        if experiment in report:
            print_experiment_summary(experiment, report[experiment])


if __name__ == "__main__":
    main()
