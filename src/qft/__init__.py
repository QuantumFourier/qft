"""Public package interface for the QFT package."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DistributedQFTBuildResult",
    "QFTDrawResult",
    "QFTStateSnapshot",
    "analyze_distributed_costs",
    "build",
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
    "qft",
    "qft_on_amplitudes",
    "validate_shape",
]

_LAZY_EXPORTS = {
    "DistributedQFTBuildResult": ("distributed", "DistributedQFTBuildResult"),
    "QFTDrawResult": ("visualization", "QFTDrawResult"),
    "QFTStateSnapshot": ("visualization", "QFTStateSnapshot"),
    "analyze_distributed_costs": ("distributed", "analyze_distributed_costs"),
    "build_bell_pair_resource": ("distributed_blocks", "build_bell_pair_resource"),
    "build_cat_disentangler_block": ("distributed_blocks", "build_cat_disentangler_block"),
    "build_cat_entangler_block": ("distributed_blocks", "build_cat_entangler_block"),
    "build_distributed_qft": ("distributed", "build_distributed_qft"),
    "build_multidimensional_qft": ("multidimensional", "build_multidimensional_qft"),
    "build_node_mapping": ("distributed", "build_node_mapping"),
    "build_nonlocal_controlled_phase_demo": ("distributed_blocks", "build_nonlocal_controlled_phase_demo"),
    "build_recursive_qft": ("standard_qft", "build_recursive_qft"),
    "build_standard_qft": ("standard_qft", "build_standard_qft"),
    "build_teleportation_leg_demo": ("distributed_blocks", "build_teleportation_leg_demo"),
    "build_teleportation_swap_demo": ("distributed_blocks", "build_teleportation_swap_demo"),
    "choose_best_method": ("distributed", "choose_best_method"),
    "dft_amplitudes": ("standard_qft", "dft_amplitudes"),
    "dimension_qubit_widths": ("multidimensional", "dimension_qubit_widths"),
    "distributed_chip_layout": ("distributed", "distributed_chip_layout"),
    "draw_qft": ("visualization", "draw_qft"),
    "expected_multidimensional_qft_state": ("multidimensional", "expected_multidimensional_qft_state"),
    "logical_to_physical_mapping": ("distributed", "logical_to_physical_mapping"),
    "padded_shape": ("multidimensional", "padded_shape"),
    "prepare_multidimensional_input": ("multidimensional", "prepare_multidimensional_input"),
    "qft_on_amplitudes": ("standard_qft", "qft_on_amplitudes"),
    "validate_shape": ("multidimensional", "validate_shape"),
}

_LAZY_MODULES = {
    "distributed": "distributed",
    "distributed_blocks": "distributed_blocks",
    "multidimensional": "multidimensional",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        module = import_module(f".{module_name}", __name__)
        value = getattr(module, attribute_name)
        globals()[name] = value
        return value

    if name in _LAZY_MODULES:
        module = import_module(f".{_LAZY_MODULES[name]}", __name__)
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def qft(
    num_qubits: int,
    do_swap: bool = True,
    recursive: bool = False,
):
    """Build a QFT circuit from the public package namespace."""
    from .standard_qft import build_standard_qft

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
        from .multidimensional import build_multidimensional_qft

        return build_multidimensional_qft(shape, method=method)

    if kind == "distributed":
        if "num_qubits" not in kwargs:
            raise TypeError("kind='distributed' requires num_qubits")
        num_qubits = kwargs.pop("num_qubits")
        num_nodes = kwargs.pop("num_nodes", 2)
        strategy = kwargs.pop("strategy", "contiguous")
        shots = kwargs.pop("shots", 2048)
        _reject_unexpected_arguments(kind, kwargs)
        from .distributed import build_distributed_qft

        return build_distributed_qft(
            num_qubits,
            num_nodes=num_nodes,
            strategy=strategy,
            shots=shots,
        )

    raise ValueError("kind must be 'standard', 'multidimensional', or 'distributed'")
