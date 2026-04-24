"""Recursive Quantum Fourier Transform utilities."""

from .standard_recursive_qft import build_recursive_qft


def qft(num_qubits: int):
    """Convenience alias for building the recursive QFT circuit."""
    return build_recursive_qft(num_qubits=num_qubits)
