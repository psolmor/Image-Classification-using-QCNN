import benchmark

# Configuración del benchmark
dataset = "fashion_mnist"  # 'fashion_mnist' o "mnist"
class_pairs = [(0,1)]# [(0, 1),(2,3)...]
unitary_circuit = ["CONV"]  # "TTN" o "CONV"
resize = ["autoencoder"]  # "area", "bilinear" o "nearest"
embedding = "angle" if "autoencoder" in resize else "amplitude"

benchmark.benchmark(class_pairs, unitary_circuit, embedding, resize, dataset,iterations=5,loss="mse")