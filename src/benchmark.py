import data
import training
import testing
import circuit
import matplotlib.pyplot as plt
import time


def benchmark(number_pairs):

    results_file = "results.txt"

    with open(results_file, "w") as f:
        for number1, number2 in number_pairs:
            start_time = time.time()
            
            x_train, x_test, y_train, y_test = data.data_load_and_process(number1, number2)
            
            print(f"Training Model for numbers {number1} and {number2}...")
            loss_history, trained_params = training.circuit_training(x_train, y_train)

            print("Testing...")
            predictions = [circuit.QCNN(x, trained_params) for x in x_test]
            test_accuracy = testing.accuracy_test(predictions, y_test)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            f.write(f"Numbers: {number1} and {number2}\n")
            f.write(f"Time elapsed: {elapsed_time:.2f} seconds\n")
            f.write(f"Test accuracy: {test_accuracy * 100:.2f}%\n\n")
            
            print(f"Numbers: {number1} and {number2}")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print(f"Test accuracy: {test_accuracy * 100:.2f}%\n")

    print("Results saved to", results_file)
