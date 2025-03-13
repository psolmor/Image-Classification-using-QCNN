import pennylane as qml

def data_embedding(X,embedding):

    if embedding=="Amplitude":
        #print("Amplitude Embedding in progress..")
        qml.AmplitudeEmbedding(X,wires=range(8),normalize=True)
        #print("Amplitude Embedding done")
    elif embedding=="Angle":
        #print("Angle Embedding in progress..")
        qml.AngleEmbedding(X, wires=range(8), rotation="Y")
        #print("Angle Embedding done")

