import components 
import pennylane as qml
import embedding as emb


#conv layers
def conv_layer1(U, params):
    U(params, wires=[0, 7])
    for i in range(0, 8, 2):
        U(params, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(params, wires=[i, i + 1])

def conv_layer2(U, params):
    U(params, wires=[0, 6])

    U(params, wires=[0, 2])
    U(params, wires=[4, 6])
    U(params, wires=[2, 4])

def conv_layer3(U, params):
    U(params, wires=[0,4])

# Pooling layers
def pooling_layer1(V, params):
    for i in range(0, 8, 2):
        V(params, wires=[i + 1, i])

def pooling_layer2(V, params):
    V(params, wires=[2,0])
    V(params, wires=[6,4])
    
def pooling_layer3(V, params):
    V(params, wires=[0,4])

#whole circuit structure
def QCNN_structure(U,params,U_params):
    
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 3 * U_params + 2]
    param5 = params[3 * U_params + 2: 3 * U_params + 4]
    param6 = params[3 * U_params + 4: 3 * U_params + 6]

    conv_layer1(U, param1)
    pooling_layer1(components.pooling_circuit, param4)
    conv_layer2(U, param2)
    pooling_layer2(components.pooling_circuit, param5)
    conv_layer3(U, param3)
    pooling_layer3(components.pooling_circuit, param6)


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