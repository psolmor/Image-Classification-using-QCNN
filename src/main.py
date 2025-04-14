import benchmark
import utils

#Here the nechmark is set up by hardcodding the desired parameters

dataset="mnist" # 'fashion_mnist' "mnist"
class_pairs = [(0, 1)] # [(0, 1),(2,3)...]
unitary_circuit=["TTN"] # "TTN" "CONV"
resize=["area","bilinear","nearest"] # "area" "bilinear" "nearest"
params_num=utils.param_num(resize) # TTN=12 CONV=15
embedding="Amplitude"

benchmark.benchmark(class_pairs,unitary_circuit,embedding,resize,dataset)
