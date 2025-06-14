# benchmark.py

import data
import training
import testing
import circuit
import time
import utils
import numpy as np  # Necesario para cálculos estadísticos

def benchmark(class_pairs, unitary_arr, embedding, resize, dataset, iterations,loss):
    results_file = "results.txt"
    num_iterations = iterations

    with open(results_file, "a") as f:
        f.write("#################################\n")
        f.write(f"{time.ctime()}\n\n")
        for unitary in unitary_arr:
            for class1, class2 in class_pairs:
                for resizing_method in resize:

                    print(f"Parameters: {utils.get_label(class1, dataset)} "
                          f"{utils.get_label(class2, dataset)}, unitary {unitary}, "
                          f"embedding {embedding}, resize {resizing_method}")

                    training_times = []
                    test_accuracies = []

                    for iteration in range(num_iterations):
                        x_train, x_test, y_train, y_test = data.data_load_and_process(
                            class1, class2, resizing_method, loss,dataset
                        )

                        start_time = time.time()
                        trained_params = training.circuit_training(
                            x_train, y_train, unitary, embedding,loss
                        )
                        end_time = time.time()
                        training_times.append(end_time - start_time)

                        print("Testing...")
                        predictions = [
                            circuit.QCNN(x, trained_params, unitary, embedding,loss)
                            for x in x_test
                        ]
                        test_accuracy = testing.accuracy_test(predictions, y_test)
                        test_accuracies.append(test_accuracy)

                        print(f"Iteration {iteration + 1}: "
                              f"Time Training: {(end_time - start_time):.2f} seconds, "
                              f"Test accuracy: {test_accuracy * 100:.2f}%")

                    avg_time = np.mean(training_times)
                    std_time = np.std(training_times)
                    avg_accuracy = np.mean(test_accuracies)
                    std_accuracy = np.std(test_accuracies)

                    f.write(f"Parameters:\n{utils.get_label(class1, dataset)} "
                            f"{utils.get_label(class2, dataset)}\n")
                    f.write(f"Unitary: {unitary}\n")
                    f.write(f"Number_parameters: {utils.param_num(unitary)}\n")
                    f.write(f"Embedding: {embedding}\n")
                    f.write(f"Resize: {resizing_method}\n")
                    f.write(f"Training Time: {avg_time:.2f} ± {std_time:.2f} seconds "
                            f"(mean ± std)\n")
                    f.write(f"Test accuracy: {avg_accuracy * 100:.2f}% ± "
                            f"{std_accuracy * 100:.2f}% (mean ± std)\n\n")

                    print("\nSummary for this configuration:")
                    print(f"Average Training Time: {avg_time:.2f} ± {std_time:.2f} seconds")
                    print(f"Average Test Accuracy: {avg_accuracy * 100:.2f}% ± "
                          f"{std_accuracy * 100:.2f}%\n")

    print("Results saved to", results_file)
