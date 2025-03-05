import data
import training
import benchmark
import circuit
import numpy as np
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = data.data_load_and_process("mnist")

U = "U_TTN"  
U_params = 2  


loss_history, trained_params = training.circuit_training(x_train, y_train, U, U_params, circuit.QCNN)

plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Evolution cost function")
plt.show()

predictions = [circuit.QCNN(x, trained_params, U, U_params) for x in x_test]
test_accuracy = benchmark.accuracy_test(predictions, y_test)
print(f"Accuracy of test set: {test_accuracy * 100:.2f}%")


