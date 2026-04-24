from __future__ import annotations

import sys
from pathlib import Path

import pytest

PARENT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PARENT_DIR / "src"
for candidate in (PARENT_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from distributed_qft_blocks import (
    build_bell_pair_resource,
    build_cat_disentangler_block,
    build_cat_entangler_block,
    build_nonlocal_controlled_phase_demo,
    build_teleportation_leg_demo,
    build_teleportation_swap_demo,
)
from distributed_qft_comparison import (
    analyze_distributed_costs,
    build_node_mapping,
    choose_best_method,
    distributed_chip_layout,
    logical_to_physical_mapping,
)
from qft.standard_recursive_qft import build_standard_qft


def test_build_node_mapping_contiguous_assigns_balanced_blocks() -> None:
    assert build_node_mapping(6, 3, "contiguous") == {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
    }


def test_build_node_mapping_interleaved_round_robins_qubits() -> None:
    assert build_node_mapping(6, 3, "interleaved") == {
        0: 0,
        1: 1,
        2: 2,
        3: 0,
        4: 1,
        5: 2,
    }


def test_build_node_mapping_rejects_invalid_node_counts() -> None:
    with pytest.raises(ValueError):
        build_node_mapping(4, 0, "contiguous")

    with pytest.raises(ValueError):
        build_node_mapping(2, 3, "contiguous")


def test_distributed_chip_layout_matches_documented_capacity() -> None:
    layout = distributed_chip_layout()
    assert layout["backend_name"] == "fake_fez"
    assert len(layout["nodes"]) == 3
    assert sum(node["capacity"] for node in layout["nodes"]) == 150


def test_logical_to_physical_mapping_assigns_slots_within_each_node() -> None:
    mapping = logical_to_physical_mapping({0: 1, 1: 1, 2: 0})
    assert mapping == {0: 50, 1: 51, 2: 0}


def test_analyze_distributed_costs_identifies_nonlocal_work() -> None:
    node_mapping = build_node_mapping(4, 2, "contiguous")
    report = analyze_distributed_costs(build_standard_qft(4), node_mapping)

    assert report["nonlocal_gate_total"] > 0
    assert report["shared_entangled_pairs"] > 0
    assert any(entry["gate"] == "swap" for entry in report["execution_log"] if entry["locality"] == "non-local")


def test_choose_best_method_prefers_lower_communication_cost() -> None:
    reports = [
        {
            "method": "A",
            "shared_entangled_pairs": 4,
            "classical_bits": 8,
            "teleportations": 2,
            "nonlocal_gate_total": 3,
            "total_depth": 10,
            "total_gate_count": 20,
        },
        {
            "method": "B",
            "shared_entangled_pairs": 2,
            "classical_bits": 8,
            "teleportations": 2,
            "nonlocal_gate_total": 3,
            "total_depth": 10,
            "total_gate_count": 20,
        },
    ]

    assert choose_best_method(reports)["method"] == "B"


@pytest.mark.parametrize(
    ("builder", "expected_qubits", "expected_clbits", "required_ops"),
    (
        (build_bell_pair_resource, 2, 0, {"h": 1, "cx": 1}),
        (build_cat_entangler_block, 3, 1, {"measure": 1, "if_else": 1}),
        (build_cat_disentangler_block, 2, 1, {"measure": 1, "if_else": 1}),
        (build_nonlocal_controlled_phase_demo, 4, 2, {"cp": 1, "measure": 2}),
        (build_teleportation_leg_demo, 3, 2, {"measure": 2, "if_else": 2}),
        (build_teleportation_swap_demo, 4, 4, {"measure": 4, "if_else": 4}),
    ),
)
def test_distributed_block_builders_create_expected_resources(builder, expected_qubits, expected_clbits, required_ops) -> None:
    circuit = builder()
    assert circuit.num_qubits == expected_qubits
    assert circuit.num_clbits == expected_clbits

    operations = circuit.count_ops()
    for name, count in required_ops.items():
        assert operations.get(name, 0) == count
