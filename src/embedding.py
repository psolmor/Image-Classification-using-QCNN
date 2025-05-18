from pennylane.templates.embeddings import AmplitudeEmbedding,AngleEmbedding

def data_embedding(X,embedding):

    if embedding=="Amplitude":
        AmplitudeEmbedding(X,wires=range(8),normalize=True)
    if embedding == "Angle":
        AngleEmbedding(X, wires=range(8), rotation='Y')


