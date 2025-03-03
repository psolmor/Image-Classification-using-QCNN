import pennylane as qml

def data_embedding(X):
    print("Amplitude Embedding in progress..")
    qml.AmplitudeEmbedding(X,wires=range(8),normalize=True)
    print("Amplitude Embedding done")

