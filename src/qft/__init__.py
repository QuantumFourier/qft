"""Public package interface for the initial QFT library release."""

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


from . import recursive, standard

__all__ = [
    "QFTDrawResult",
    "QFTStateSnapshot",
    "build_recursive_qft",
    "build_standard_qft",
    "dft_amplitudes",
    "draw_qft",
    "qft",
    "qft_on_amplitudes",
    "recursive",
    "standard",
]
