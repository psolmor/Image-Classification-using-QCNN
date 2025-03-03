import pennylane as qml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import data 
import embedding

dev = qml.device("default.qubit", wires=8)


@qml.qnode(dev)
def circuit(f=None):
    embedding.data_embedding(f)
    return qml.state()


dataxd = data.data_load_and_process()
image = dataxd[0][0]


state = circuit(f=image)


plt.figure(figsize=(10, 5))
plt.bar(range(len(state)), np.abs(state) ** 2)
plt.xlabel("Estado computacional |i⟩")
plt.ylabel("Probabilidad")
plt.title("Distribución de probabilidades después de Amplitude Embedding")
plt.show()

