from pennylane.templates.embeddings import AmplitudeEmbedding,AngleEmbedding

def data_embedding(X, embedding):
    emb = embedding.lower() 
    if emb == "amplitude":
        AmplitudeEmbedding(X, wires=range(8), normalize=True)
    elif emb == "angle":
        AngleEmbedding(X, wires=range(8), rotation='Y')
    else:
        raise ValueError(f"Embedding desconocido: {embedding}")


