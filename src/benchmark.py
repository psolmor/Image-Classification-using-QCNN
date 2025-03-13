import data
import training
import testing
import circuit
import matplotlib.pyplot as plt
import time


def benchmark(number_pairs,params_num,unitary_arr,embedding):

    results_file = "results.txt"

    with open(results_file, "w") as f:
        for unitary in unitary_arr:
            for number1, number2 in number_pairs:
                start_time = time.time()
                print(f"Parameters: numbers {number1} and {number2}, unitary {unitary}, embedding {embedding} ")
                x_train, x_test, y_train, y_test = data.data_load_and_process(number1, number2)
            
                trained_params = training.circuit_training(x_train, y_train,params_num,unitary,embedding)
                end_time = time.time()
                elapsed_time = end_time - start_time

                print("Testing...")
                predictions = [circuit.QCNN(x, trained_params,unitary,embedding) for x in x_test]
                test_accuracy = testing.accuracy_test(predictions, y_test)
                

                
                f.write(f"Parameters: numbers {number1} and {number2}, unitary {unitary}, embedding {embedding}\n")
                f.write(f"Unitary: {unitary}\n")
                f.write(f"Training Time: {elapsed_time:.2f} seconds\n")
                f.write(f"Test accuracy: {test_accuracy * 100:.2f}%\n\n")
                
                print(f"Numbers: {number1} and {number2}")
                print(f"Time Training: {elapsed_time:.2f} seconds")
                print(f"Test accuracy: {test_accuracy * 100:.2f}%\n")

    print("Results saved to", results_file)
