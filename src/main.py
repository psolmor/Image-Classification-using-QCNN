import benchmark

# Configuración del benchmark
dataset = "mnist"  # 'fashion_mnist' o "mnist"
class_pairs = [(3,8)]  # [(0, 1),(2,3)...]
unitary_circuit = ["CONV"]  # "TTN" o "CONV"
resize = ["bilinear","nearest"  ]  # "area", "bilinear" o "nearest"
embedding = "Amplitude"  # "Amplitude"

benchmark.benchmark(class_pairs, unitary_circuit, embedding, resize, dataset,iterations=5)