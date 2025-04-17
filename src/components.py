import pennylane as qml
import numpy as np

def conv_circuit(params, wires):
    """Convolutional circuit (3 parameters)"""
    if len(params) < 3:
        raise ValueError(f"conv_circuit needs 3 params, got {len(params)}")
    
    qml.RZ(params[0], wires=wires[0])
    qml.RZ(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RZ(params[2], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

def U_TTN(params, wires):
    """TTN unitary (2 parameters)"""
    if len(params) < 2:
        raise ValueError(f"U_TTN needs 2 params, got {len(params)}")
    
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

def pool_circuit(params, wires):
    """Pooling circuit (3 parameters)"""
    if len(params) < 3:
        raise ValueError(f"pool_circuit needs 3 params, got {len(params)}")
    
    qml.RZ(-np.pi/2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[1])


