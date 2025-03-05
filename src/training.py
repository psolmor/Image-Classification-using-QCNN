import circuit
import pennylane as qml
import numpy as np
import autograd.numpy as anp

def square_loss(labels,predictions):
    loss=0
    for l,p in zip(labels,predictions):
        loss+=(1-p)**2
    loss=loss/len(labels)
    return loss