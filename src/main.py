import benchmark

# Configuración del benchmark
dataset = "mnist"  # 'fashion_mnist' o "mnist"
class_pairs = [(0,1),(3,4),(4,8)]  # [(0, 1),(2,3)...]
unitary_circuit = ["CONV","TTN"]  # "TTN" o "CONV"
resize = ["bilinear"]  # "area", "bilinear" o "nearest"
embedding = "Amplitude"  # "Amplitude"

benchmark.benchmark(class_pairs, unitary_circuit, embedding, resize, dataset)