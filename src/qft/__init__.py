"""Public package interface for the initial QFT library release."""

from . import distributed, distributed_blocks, multidimensional, recursive, standard
from .distributed import (
    DistributedQFTBuildResult,
    analyze_distributed_costs,
    build_distributed_qft,
    build_node_mapping,
    choose_best_method,
    distributed_chip_layout,
    logical_to_physical_mapping,
)
from .distributed_blocks import (
    build_bell_pair_resource,
    build_cat_disentangler_block,
    build_cat_entangler_block,
    build_nonlocal_controlled_phase_demo,
    build_teleportation_leg_demo,
    build_teleportation_swap_demo,
)
from .multidimensional import (
    build_multidimensional_qft,
    dimension_qubit_widths,
    expected_multidimensional_qft_state,
    padded_shape,
    prepare_multidimensional_input,
    validate_shape,
)
from .standard_qft import (
    build_recursive_qft,
    build_standard_qft,
    dft_amplitudes,
    qft_on_amplitudes,
)
from .visualization import QFTDrawResult, QFTStateSnapshot, draw_qft


def qft(
    num_qubits: int,
    do_swap: bool = True,
    recursive: bool = False,
):
    """Build a QFT circuit from the public package namespace."""
    return build_standard_qft(
        num_qubits=num_qubits,
        do_swaps=do_swap,
        recursive=recursive,
    )


def _reject_unexpected_arguments(kind: str, arguments: dict) -> None:
    if arguments:
        unexpected = ", ".join(sorted(arguments))
        raise TypeError(f"Unexpected arguments for kind={kind!r}: {unexpected}")


def build(*, kind: str = "standard", **kwargs):
    """Dispatch to the main package builders through one entry point."""
    if kind == "standard":
        if "num_qubits" not in kwargs:
            raise TypeError("kind='standard' requires num_qubits")
        num_qubits = kwargs.pop("num_qubits")
        do_swap = kwargs.pop("do_swap", True)
        recursive = kwargs.pop("recursive", False)
        _reject_unexpected_arguments(kind, kwargs)
        return qft(num_qubits, do_swap=do_swap, recursive=recursive)

    if kind == "multidimensional":
        if "shape" not in kwargs:
            raise TypeError("kind='multidimensional' requires shape")
        shape = kwargs.pop("shape")
        method = kwargs.pop("method", "standard")
        _reject_unexpected_arguments(kind, kwargs)
        return build_multidimensional_qft(shape, method=method)

    if kind == "distributed":
        if "num_qubits" not in kwargs:
            raise TypeError("kind='distributed' requires num_qubits")
        num_qubits = kwargs.pop("num_qubits")
        num_nodes = kwargs.pop("num_nodes", 2)
        strategy = kwargs.pop("strategy", "contiguous")
        shots = kwargs.pop("shots", 2048)
        _reject_unexpected_arguments(kind, kwargs)
        return build_distributed_qft(
            num_qubits,
            num_nodes=num_nodes,
            strategy=strategy,
            shots=shots,
        )

    raise ValueError("kind must be 'standard', 'multidimensional', or 'distributed'")

__all__ = [
    "DistributedQFTBuildResult",
    "QFTDrawResult",
    "QFTStateSnapshot",
    "analyze_distributed_costs",
    "build_bell_pair_resource",
    "build_cat_disentangler_block",
    "build_cat_entangler_block",
    "build_distributed_qft",
    "build_multidimensional_qft",
    "build_node_mapping",
    "build_nonlocal_controlled_phase_demo",
    "build_recursive_qft",
    "build_standard_qft",
    "build_teleportation_leg_demo",
    "build_teleportation_swap_demo",
    "choose_best_method",
    "dft_amplitudes",
    "dimension_qubit_widths",
    "distributed",
    "distributed_blocks",
    "distributed_chip_layout",
    "draw_qft",
    "expected_multidimensional_qft_state",
    "logical_to_physical_mapping",
    "multidimensional",
    "padded_shape",
    "prepare_multidimensional_input",
    "build",
    "qft",
    "qft_on_amplitudes",
    "recursive",
    "standard",
    "validate_shape",
]
