# circuit.py

import components
import utils
import embedding as emb
import pennylane as qml

dev = qml.device("default.qubit", wires=8)
@qml.qnode(dev)
def QCNN(X, params, unitary, embedding):
    emb.data_embedding(X, embedding)

    if unitary == "TTN":
        unitary_func = components.U_TTN
        params_per_unitary = 2
    elif unitary == "CONV":
        unitary_func = components.conv_circuit
        params_per_unitary = 3
    else:
        raise ValueError("Unitary must be 'CONV' or 'TTN'")
    
    QCNN_structure(params, unitary_func, params_per_unitary)
    return qml.expval(qml.PauliZ(7))


def QCNN_structure(params, unitary_func, params_per_unitary):
    param_index = 0

    # First Convolutional Layer (8 qubits)
    for i in range(0, 8, 2):
        unitary_func(params[param_index:param_index+params_per_unitary], [i, i+1])
        param_index += params_per_unitary
    for i in range(1, 7, 2):
        unitary_func(params[param_index:param_index+params_per_unitary], [i, i+1])
        param_index += params_per_unitary
    unitary_func(params[param_index:param_index+params_per_unitary], [7, 0])
    param_index += params_per_unitary

    # First Pooling Layer (8→4 qubits)
    for i in range(4):
        components.pool_circuit(params[param_index:param_index+3], [i, i+4])
        param_index += 3

    # Second Convolutional Layer (4 qubits)
    for i in range(4, 8, 2):
        unitary_func(params[param_index:param_index+params_per_unitary], [i, i+1])
        param_index += params_per_unitary
    unitary_func(params[param_index:param_index+params_per_unitary], [7, 4])
    param_index += params_per_unitary

    # Second Pooling Layer (4→2 qubits)
    components.pool_circuit(params[param_index:param_index+3], [4, 6])
    param_index += 3
    components.pool_circuit(params[param_index:param_index+3], [5, 7])
    param_index += 3

    # Third Convolutional Layer (2 qubits)
    unitary_func(params[param_index:param_index+params_per_unitary], [6, 7])
    param_index += params_per_unitary

    # Final Pooling (2→1 qubit)
    components.pool_circuit(params[param_index:param_index+3], [6, 7])