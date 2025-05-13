from pennylane.templates.embeddings import AmplitudeEmbedding,AngleEmbedding

def data_embedding(X,embedding):

    if embedding=="Amplitude":
        #print("Amplitude Embedding in progress..")
        AmplitudeEmbedding(X,wires=range(8),normalize=True)
        #print("Amplitude Embedding done")


