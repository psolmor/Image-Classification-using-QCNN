import pennylane as qml
import numpy as np

def conv_circuit(params, wires):
    """Convolutional Circuit #3 params"""
    qml.RZ(params[0], wires=wires[0])
    qml.RZ(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RZ(params[2], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

def pool_circuit(params, wires):
    """Pooling Circuit #3 params"""
    qml.RZ(-np.pi/2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[1])

def U_TTN(params, wires):
    """Simplified Convolutional Circuit #3 params"""
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


