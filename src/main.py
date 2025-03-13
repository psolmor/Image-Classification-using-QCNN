import benchmark


number_pairs = [(0, 1),(2,3)] # [(0, 1),(2,3)...]
unitary_circuit=["CONV","TTN"] # "TTN" "CONV"
params_num=15 # TTN=12 CONV=15
embedding="Amplitude"

benchmark.benchmark(number_pairs,params_num,unitary_circuit,embedding)
