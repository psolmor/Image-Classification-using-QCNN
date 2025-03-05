import components 
import embedding
import pennylane as qml

def QCNN_structure(U,params,U_params):
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 3 * U_params + 2]
    param5 = params[3 * U_params + 2: 3 * U_params + 4]
    param6 = params[3 * U_params + 4: 3 * U_params + 6]

    # Pooling Ansatz1 is used by default
    components.conv_layer1(U, param1)
    components.pooling_layer1(components.pooling_circuit, param4)
    components.conv_layer2(U, param2)
    components.pooling_layer2(components.pooling_circuit, param5)
    components.conv_layer3(U, param3)
    components.pooling_layer3(components.pooling_circuit, param6)


dev = qml.device('default.qubit', wires = 8)
@qml.qnode(dev)
def QCNN(X, params, U, U_params, embedding_type='Amplitude', cost_fn='mse'):


    # Data Embedding
    embedding.data_embedding(X, embedding_type=embedding_type)

    # Quantum Convolutional Neural Network
    if U == 'U_TTN':
        QCNN_structure(components.U_TTN, params, U_params)
    else:
        print("Invalid Unitary Ansatze")
        return False

    result = qml.expval(qml.PauliZ(4))

    return result