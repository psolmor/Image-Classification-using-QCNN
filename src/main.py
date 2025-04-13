import benchmark

#Here the nechmark is set up by hardcodding the desired parameters

number_pairs = [(0, 1)] # [(0, 1),(2,3)...]
unitary_circuit=["CONV"] # "TTN" "CONV"
resize=["area","bilinear","nearest"] # "area" "bilinear" "nearest"
params_num=15 # TTN=12 CONV=15
embedding="Amplitude"

benchmark.benchmark(number_pairs,unitary_circuit,embedding,resize)
