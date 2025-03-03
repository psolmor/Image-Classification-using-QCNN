import pennylane as qml
import numpy as np 

dev = qml.device('default.qubit', wires=2)


def Pooling_ansatz1(params, wires): #2 params
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])


@qml.qnode(dev)
def circuit(f=None):
    qml.AmplitudeEmbedding(features=f, wires=range(2))
    Pooling_ansatz1([np.pi,np.pi/2],range(2))
    return qml.state()

state = circuit(f=[1/2, 1/2, 1/2, 1/2])    
print(state)