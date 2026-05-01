from __future__ import annotations

from math import pi

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


# Shared Bell-pair resource used between two nodes.
def build_bell_pair_resource() -> QuantumCircuit:
    left = QuantumRegister(1, "node_a_chan")
    right = QuantumRegister(1, "node_b_chan")
    circuit = QuantumCircuit(left, right, name="BellPair")
    circuit.h(left[0])
    circuit.cx(left[0], right[0])
    return circuit


# Illustrative cat-entangler step that makes remote control available on node B.
def build_cat_entangler_block() -> QuantumCircuit:
    control = QuantumRegister(1, "control")
    left = QuantumRegister(1, "node_a_chan")
    right = QuantumRegister(1, "node_b_chan")
    classical = ClassicalRegister(1, "m_ent")
    circuit = QuantumCircuit(control, left, right, classical, name="CatEntangler")

    circuit.h(left[0])
    circuit.cx(left[0], right[0])
    circuit.cx(control[0], left[0])
    circuit.measure(left[0], classical[0])
    with circuit.if_test((classical, 1)):
        circuit.x(right[0])

    return circuit


# Illustrative cat-disentangler step that returns the distributed control to node A.
def build_cat_disentangler_block() -> QuantumCircuit:
    control = QuantumRegister(1, "control")
    right = QuantumRegister(1, "node_b_chan")
    classical = ClassicalRegister(1, "m_dis")
    circuit = QuantumCircuit(control, right, classical, name="CatDisentangler")

    circuit.h(right[0])
    circuit.measure(right[0], classical[0])
    with circuit.if_test((classical, 1)):
        circuit.z(control[0])

    return circuit


# Entanglement-assisted non-local controlled-phase gate for visualization.
def build_nonlocal_controlled_phase_demo(angle: float = pi / 4) -> QuantumCircuit:
    control = QuantumRegister(1, "control_a")
    left = QuantumRegister(1, "chan_a")
    right = QuantumRegister(1, "chan_b")
    target = QuantumRegister(1, "target_b")
    ent_classical = ClassicalRegister(1, "m_ent")
    dis_classical = ClassicalRegister(1, "m_dis")
    circuit = QuantumCircuit(
        control,
        left,
        right,
        target,
        ent_classical,
        dis_classical,
        name="NonLocalCP",
    )

    # Create the shared Bell pair between the two nodes.
    circuit.h(left[0])
    circuit.cx(left[0], right[0])
    circuit.barrier()

    # Cat-entangler: push the control information onto the remote channel qubit.
    circuit.cx(control[0], left[0])
    circuit.measure(left[0], ent_classical[0])
    with circuit.if_test((ent_classical, 1)):
        circuit.x(right[0])
    circuit.barrier()

    # Apply the controlled phase locally on node B.
    circuit.cp(angle, right[0], target[0])
    circuit.barrier()

    # Cat-disentangler: clean up the remote control and restore the original control state.
    circuit.h(right[0])
    circuit.measure(right[0], dis_classical[0])
    with circuit.if_test((dis_classical, 1)):
        circuit.z(control[0])

    return circuit


# One standard teleportation leg, used as the building block of a distributed swap.
def build_teleportation_leg_demo() -> QuantumCircuit:
    source = QuantumRegister(1, "source_a")
    left = QuantumRegister(1, "chan_a")
    right = QuantumRegister(1, "recv_b")
    classical = ClassicalRegister(2, "m")
    circuit = QuantumCircuit(source, left, right, classical, name="Teleport")

    circuit.h(left[0])
    circuit.cx(left[0], right[0])
    circuit.barrier()

    circuit.cx(source[0], left[0])
    circuit.h(source[0])
    circuit.measure(source[0], classical[0])
    circuit.measure(left[0], classical[1])

    with circuit.if_test((classical[1], 1)):
        circuit.x(right[0])
    with circuit.if_test((classical[0], 1)):
        circuit.z(right[0])

    return circuit


# Illustrative distributed SWAP built from two teleportation legs.
def build_teleportation_swap_demo() -> QuantumCircuit:
    left_data = QuantumRegister(1, "left_data")
    left_chan = QuantumRegister(1, "left_chan")
    right_chan = QuantumRegister(1, "right_chan")
    right_data = QuantumRegister(1, "right_data")
    classical = ClassicalRegister(4, "m")
    circuit = QuantumCircuit(
        left_data,
        left_chan,
        right_chan,
        right_data,
        classical,
        name="TeleportSwap",
    )

    # Teleport the left data qubit to the right side.
    circuit.h(left_chan[0])
    circuit.cx(left_chan[0], right_chan[0])
    circuit.cx(left_data[0], left_chan[0])
    circuit.h(left_data[0])
    circuit.measure(left_data[0], classical[0])
    circuit.measure(left_chan[0], classical[1])
    with circuit.if_test((classical[1], 1)):
        circuit.x(right_chan[0])
    with circuit.if_test((classical[0], 1)):
        circuit.z(right_chan[0])
    circuit.barrier()

    # Teleport the right data qubit back to the left side.
    circuit.h(right_data[0])
    circuit.cx(right_data[0], left_data[0])
    circuit.cx(right_chan[0], right_data[0])
    circuit.h(right_chan[0])
    circuit.measure(right_chan[0], classical[2])
    circuit.measure(right_data[0], classical[3])
    with circuit.if_test((classical[3], 1)):
        circuit.x(left_data[0])
    with circuit.if_test((classical[2], 1)):
        circuit.z(left_data[0])

    return circuit
