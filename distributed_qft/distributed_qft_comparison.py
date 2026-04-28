from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeFez

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
SRC_DIR = PARENT_DIR / "src"
for candidate in (PARENT_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from qft.standard_qft import build_recursive_qft, build_standard_qft
from qft_sampler_utils import (
    build_measured_qft_circuit,
    build_sample_amplitudes,
    counts_summary,
    sample_aer_counts,
    sample_noisy_aer_counts,
    total_variation_distance,
)

NODE_CAPACITY = 50
TOTAL_NODES = 3
UNUSED_PHYSICAL_QUBITS = 6


# Split qubits across nodes using a simple placement rule.
def build_node_mapping(num_qubits: int, num_nodes: int, strategy: str) -> dict[int, int]:
    if num_nodes < 1 or num_nodes > TOTAL_NODES:
        raise ValueError(f"num_nodes must be between 1 and {TOTAL_NODES} for the 156-qubit chip layout.")
    if num_nodes > num_qubits:
        raise ValueError("num_nodes must not be greater than num_qubits.")

    if strategy == "interleaved":
        return {qubit: qubit % num_nodes for qubit in range(num_qubits)}

    base_width = num_qubits // num_nodes
    extra = num_qubits % num_nodes
    mapping: dict[int, int] = {}
    qubit = 0

    for node in range(num_nodes):
        width = base_width + (1 if node < extra else 0)
        for _ in range(width):
            mapping[qubit] = node
            qubit += 1

    return mapping


# Describe the fixed three-node layout on the 156-qubit chip.
def distributed_chip_layout() -> dict:
    nodes = []
    for node in range(TOTAL_NODES):
        start = node * NODE_CAPACITY
        end = start + NODE_CAPACITY - 1
        nodes.append(
            {
                "node": node,
                "physical_qubit_range": [start, end],
                "capacity": NODE_CAPACITY,
            }
        )

    return {
        "backend_name": "fake_fez",
        "total_physical_qubits": TOTAL_NODES * NODE_CAPACITY + UNUSED_PHYSICAL_QUBITS,
        "nodes": nodes,
        "unused_physical_qubit_range": [TOTAL_NODES * NODE_CAPACITY, TOTAL_NODES * NODE_CAPACITY + UNUSED_PHYSICAL_QUBITS - 1],
    }


# Map each logical qubit to a physical qubit slot within its assigned node.
def logical_to_physical_mapping(node_mapping: dict[int, int]) -> dict[int, int]:
    usage_by_node = {node: 0 for node in range(TOTAL_NODES)}
    physical_mapping: dict[int, int] = {}

    for logical_qubit in sorted(node_mapping):
        node = node_mapping[logical_qubit]
        slot = usage_by_node[node]
        if slot >= NODE_CAPACITY:
            raise ValueError(f"Node {node} exceeds its {NODE_CAPACITY}-qubit capacity.")
        physical_mapping[logical_qubit] = node * NODE_CAPACITY + slot
        usage_by_node[node] += 1

    return physical_mapping


# Turn circuit qubits into integer indices for reporting.
def qubit_indices(circuit: QuantumCircuit, instruction) -> list[int]:
    return [circuit.find_bit(qubit).index for qubit in instruction.qubits]


# Describe one gate step in plain language for the execution log.
def build_log_entry(
    step: int,
    gate_name: str,
    indices: list[int],
    nodes: list[int],
    locality: str,
    action: str,
    new_entangled_pairs: int = 0,
    new_classical_bits: int = 0,
    new_teleportations: int = 0,
) -> dict:
    return {
        "step": step,
        "gate": gate_name,
        "qubits": indices,
        "nodes": nodes,
        "locality": locality,
        "action": action,
        "new_shared_entangled_pairs": new_entangled_pairs,
        "new_classical_bits": new_classical_bits,
        "new_teleportations": new_teleportations,
    }


# Count local and non-local work using a simple distributed cost model.
def analyze_distributed_costs(circuit: QuantumCircuit, node_mapping: dict[int, int]) -> dict:
    local_gate_counts: dict[str, int] = {}
    nonlocal_gate_counts: dict[str, int] = {}
    distributed_control_blocks = 0
    shared_entangled_pairs = 0
    classical_bits = 0
    teleportations = 0
    execution_log: list[dict] = []
    previous_control_block: tuple[int, int, int] | None = None

    for step, instruction in enumerate(circuit.data, start=1):
        gate_name = instruction.operation.name
        indices = qubit_indices(circuit, instruction)
        nodes = [node_mapping[index] for index in indices]

        if len(set(nodes)) == 1:
            local_gate_counts[gate_name] = local_gate_counts.get(gate_name, 0) + 1
            execution_log.append(
                build_log_entry(
                    step,
                    gate_name,
                    indices,
                    nodes,
                    "local",
                    f"Apply {gate_name} locally on node {nodes[0]}.",
                )
            )
            if gate_name != "cp":
                previous_control_block = None
            continue

        nonlocal_gate_counts[gate_name] = nonlocal_gate_counts.get(gate_name, 0) + 1

        if gate_name == "cp":
            control, target = indices
            control_block = (node_mapping[control], node_mapping[target], control)
            new_entangled_pairs = 0
            new_classical_bits = 0

            if control_block != previous_control_block:
                distributed_control_blocks += 1
                shared_entangled_pairs += 1
                classical_bits += 2
                new_entangled_pairs = 1
                new_classical_bits = 2
                previous_control_block = control_block
                action = (
                    f"Create a distributed control block from node {node_mapping[control]} "
                    f"to node {node_mapping[target]}, then apply cp on the target node."
                )
            else:
                action = (
                    f"Reuse the existing distributed control block from node {node_mapping[control]} "
                    f"to node {node_mapping[target]} and apply cp."
                )

            execution_log.append(
                build_log_entry(
                    step,
                    gate_name,
                    indices,
                    nodes,
                    "non-local",
                    action,
                    new_entangled_pairs=new_entangled_pairs,
                    new_classical_bits=new_classical_bits,
                )
            )
            continue

        if gate_name == "swap":
            teleportations += 2
            shared_entangled_pairs += 2
            classical_bits += 4
            execution_log.append(
                build_log_entry(
                    step,
                    gate_name,
                    indices,
                    nodes,
                    "non-local",
                    f"Implement swap between node {nodes[0]} and node {nodes[1]} using teleportation.",
                    new_entangled_pairs=2,
                    new_classical_bits=4,
                    new_teleportations=2,
                )
            )
        else:
            execution_log.append(
                build_log_entry(
                    step,
                    gate_name,
                    indices,
                    nodes,
                    "non-local",
                    f"Apply the non-local {gate_name} gate across nodes {nodes}.",
                )
            )

        previous_control_block = None

    return {
        "total_depth": circuit.depth(),
        "total_gate_count": circuit.size(),
        "local_gate_counts": local_gate_counts,
        "nonlocal_gate_counts": nonlocal_gate_counts,
        "nonlocal_gate_total": sum(nonlocal_gate_counts.values()),
        "distributed_control_blocks": distributed_control_blocks,
        "shared_entangled_pairs": shared_entangled_pairs,
        "classical_bits": classical_bits,
        "teleportations": teleportations,
        "execution_log": execution_log,
    }


# Collect one report for one QFT method.
def collect_method_report(
    label: str,
    builder,
    num_qubits: int,
    node_mapping: dict[int, int],
    amplitudes,
    backend,
    shots: int,
) -> dict:
    circuit = builder(num_qubits)
    costs = analyze_distributed_costs(circuit, node_mapping)
    measured_circuit = build_measured_qft_circuit(amplitudes, circuit)
    aer_report = {"available": False}

    try:
        ideal_aer_counts = sample_aer_counts(measured_circuit, shots=shots)
        noisy_aer_counts = sample_noisy_aer_counts(measured_circuit, backend=backend, shots=shots)
        aer_report = {
            "available": True,
            "ideal": counts_summary(ideal_aer_counts),
            "noisy": counts_summary(noisy_aer_counts),
        }
    except RuntimeError as exc:
        aer_report["message"] = str(exc)

    return {
        "method": label,
        "node_mapping": node_mapping,
        "physical_qubit_mapping": logical_to_physical_mapping(node_mapping),
        "aer": aer_report,
        **costs,
    }


# Pick the better method using communication cost first, then circuit size.
def choose_best_method(reports: list[dict]) -> dict:
    return min(
        reports,
        key=lambda item: (
            item["shared_entangled_pairs"],
            item["classical_bits"],
            item["teleportations"],
            item["nonlocal_gate_total"],
            item["total_depth"],
            item["total_gate_count"],
        ),
    )


# Print a compact summary table.
def print_summary(reports: list[dict]) -> None:
    print(
        f"{'Method':<22}{'Non-local':>12}{'Blocks':>10}"
        f"{'Ebits':>10}{'Cbits':>10}{'Teleports':>12}{'Depth':>10}"
    )
    for item in reports:
        print(
            f"{item['method']:<22}{item['nonlocal_gate_total']:>12}"
            f"{item['distributed_control_blocks']:>10}"
            f"{item['shared_entangled_pairs']:>10}"
            f"{item['classical_bits']:>10}"
            f"{item['teleportations']:>12}"
            f"{item['total_depth']:>10}"
        )


# Print the gate-by-gate steps that became non-local.
def print_nonlocal_execution_logs(reports: list[dict]) -> None:
    for item in reports:
        print(f"\nNon-local execution log for {item['method']}:")
        nonlocal_steps = [entry for entry in item["execution_log"] if entry["locality"] == "non-local"]

        if not nonlocal_steps:
            print("  None. All gates are local for this node layout.")
            continue

        for entry in nonlocal_steps:
            print(
                f"  Step {entry['step']}: {entry['gate']} on qubits {entry['qubits']} "
                f"across nodes {entry['nodes']} -> {entry['action']}"
            )


# Print the Aer summaries when Aer is available.
def print_aer_summaries(reports: list[dict]) -> None:
    for item in reports:
        print(f"\nAer summary for {item['method']}:")
        if not item["aer"]["available"]:
            print(f"  {item['aer']['message']}")
            continue

        print("  Ideal Aer top outcomes:")
        for bitstring, count in item["aer"]["ideal"]["top_outcomes"]:
            print(f"  {bitstring}: {count}")

        print("  Noisy Aer top outcomes:")
        for bitstring, count in item["aer"]["noisy"]["top_outcomes"]:
            print(f"  {bitstring}: {count}")


# Generate the distributed comparison report.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the standard QFT method and recursive QFT under a distributed cost model."
    )
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits in the QFT circuits.")
    parser.add_argument("--nodes", type=int, default=2, help="Number of distributed nodes.")
    parser.add_argument(
        "--strategy",
        choices=("contiguous", "interleaved"),
        default="contiguous",
        help="How qubits are assigned to nodes.",
    )
    parser.add_argument(
        "--output",
        default="distributed_qft_comparison.json",
        help="Output JSON file for the distributed comparison report.",
    )
    parser.add_argument(
        "--show-full-log",
        action="store_true",
        help="Print the full gate-by-gate execution log, including local gates.",
    )
    parser.add_argument("--shots", type=int, default=2048, help="Number of Aer shots to use.")
    args = parser.parse_args()

    node_mapping = build_node_mapping(args.qubits, args.nodes, args.strategy)
    amplitudes = build_sample_amplitudes(args.qubits)
    backend = FakeFez()
    chip_layout = distributed_chip_layout()
    reports = [
        collect_method_report(
            "Standard QFT method",
            build_standard_qft,
            args.qubits,
            node_mapping,
            amplitudes,
            backend,
            shots=args.shots,
        ),
        collect_method_report(
            "Recursive QFT",
            build_recursive_qft,
            args.qubits,
            node_mapping,
            amplitudes,
            backend,
            shots=args.shots,
        ),
    ]
    best_method = choose_best_method(reports)
    ideal_aer_distance = None
    noisy_aer_distance = None
    if reports[0]["aer"]["available"] and reports[1]["aer"]["available"]:
        ideal_aer_distance = total_variation_distance(
            reports[0]["aer"]["ideal"]["counts"],
            reports[1]["aer"]["ideal"]["counts"],
        )
        noisy_aer_distance = total_variation_distance(
            reports[0]["aer"]["noisy"]["counts"],
            reports[1]["aer"]["noisy"]["counts"],
        )

    report = {
        "num_qubits": args.qubits,
        "num_nodes": args.nodes,
        "partition_strategy": args.strategy,
        "backend_name": backend.name,
        "chip_layout": chip_layout,
        "reports": reports,
        "recommended_method": best_method["method"],
        "ideal_aer_tvd_between_methods": ideal_aer_distance,
        "noisy_aer_tvd_between_methods": noisy_aer_distance,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    output_path.write_text(json.dumps(report, indent=2))

    print(f"Node mapping: {node_mapping}")
    print(f"Physical mapping: {reports[0]['physical_qubit_mapping']}")
    print(f"Using fake backend: {backend.name}")
    print(f"Node layout: {[node['physical_qubit_range'] for node in chip_layout['nodes']]}, unused {chip_layout['unused_physical_qubit_range']}\n")
    print_summary(reports)
    print_aer_summaries(reports)
    if ideal_aer_distance is not None:
        print(f"\nIdeal Aer TVD between methods: {ideal_aer_distance:.4f}")
        print(f"Noisy Aer TVD between methods: {noisy_aer_distance:.4f}")
    print_nonlocal_execution_logs(reports)

    if args.show_full_log:
        for item in reports:
            print(f"\nFull execution log for {item['method']}:")
            for entry in item["execution_log"]:
                print(
                    f"  Step {entry['step']}: {entry['gate']} on qubits {entry['qubits']} "
                    f"at nodes {entry['nodes']} [{entry['locality']}] -> {entry['action']}"
                )

    print(f"\nRecommended method: {best_method['method']}")
    print(f"Saved distributed comparison to {output_path}")


if __name__ == "__main__":
    main()
