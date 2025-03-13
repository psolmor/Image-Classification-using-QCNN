import components 
import pennylane as qml
import embedding as emb

def QCNN_structure(U,params,U_params):
    
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 3 * U_params + 2]
    param5 = params[3 * U_params + 2: 3 * U_params + 4]
    param6 = params[3 * U_params + 4: 3 * U_params + 6]

    components.conv_layer1(U, param1)
    components.pooling_layer1(components.pooling_circuit, param4)
    components.conv_layer2(U, param2)
    components.pooling_layer2(components.pooling_circuit, param5)
    components.conv_layer3(U, param3)
    components.pooling_layer3(components.pooling_circuit, param6)


dev = qml.device('default.qubit', wires = 8)
@qml.qnode(dev)
def QCNN(X, params,unitary,embedding):

    
    emb.data_embedding(X,embedding)

    if unitary=="TTN":
        U_params = 2 
        QCNN_structure(components.U_TTN, params,U_params)
    elif unitary=="CONV":
        U_params = 3
        QCNN_structure(components.U_CONV, params,U_params)
        
    result = qml.expval(qml.PauliZ(4))

    return result