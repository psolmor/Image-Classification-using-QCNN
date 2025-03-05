import circuit as cir
import pennylane as qml
import numpy as np
import autograd.numpy as anp

def square_loss(labels,predictions):
    loss=0
    for l,p in zip(labels,predictions):
        loss+=(l-p)**2
    loss=loss/len(labels)
    return loss

def cost(params, X, Y, U, U_params,circuit):
  
    predictions = [cir.QCNN(x, params, U, U_params) for x in X]
    loss = square_loss(Y, predictions)

    return loss