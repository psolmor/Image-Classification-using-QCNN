import benchmark

# Configuración del benchmark
dataset = "mnist"  # 'fashion_mnist' o "mnist"
class_pairs = [(6,9)]# [(0, 1),(2,3)...]
unitary_circuit = ["CONV"]  # "TTN" o "CONV"
resize = ["autoencoder"]  # "area", "bilinear" o "nearest"
embedding = "Angle"

benchmark.benchmark(class_pairs, unitary_circuit, embedding, resize, dataset,iterations=1)