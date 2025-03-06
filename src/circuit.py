import components 
import pennylane as qml
import embedding 

def QCNN_structure(U,params):
    param1 = [params[0], params[1]]
    param2 = [params[2], params[3]]
    param3 = [params[4], params[5]]
    param4 = [params[6], params[7]]
    param5 = [params[8], params[9]]
    param6 = [params[10], params[11]]

    components.conv_layer1(U, param1)
    components.pooling_layer1(components.pooling_circuit, param4)
    components.conv_layer2(U, param2)
    components.pooling_layer2(components.pooling_circuit, param5)
    components.conv_layer3(U, param3)
    components.pooling_layer3(components.pooling_circuit, param6)


dev = qml.device('default.qubit', wires = 8)
@qml.qnode(dev)
def QCNN(X, params):

    embedding.data_embedding(X)
    QCNN_structure(components.U_TTN, params)
    result = qml.expval(qml.PauliZ(4))

    return result