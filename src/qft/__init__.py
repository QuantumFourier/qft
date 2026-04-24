"""Public package interface for the initial QFT library release."""

from .standard_recursive_qft import (
    build_recursive_qft,
    build_standard_qft,
    dft_amplitudes,
    qft_on_amplitudes,
)


def qft(num_qubits: int, do_swaps: bool = True):
    """Convenience alias for building the standard QFT circuit."""
    return build_standard_qft(num_qubits=num_qubits, do_swaps=do_swaps)


from . import recursive, standard

__all__ = [
    "build_recursive_qft",
    "build_standard_qft",
    "dft_amplitudes",
    "qft",
    "qft_on_amplitudes",
    "recursive",
    "standard",
]
