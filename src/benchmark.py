import data
import training
import testing
import circuit
import matplotlib.pyplot as plt
import time

def benchmark(class_pairs, unitary_arr, embedding,resize,dataset):
    results_file = "results.txt"

    # Abrir el archivo en modo de adición
    with open(results_file, "a") as f:
        f.write("#################################\n")
        for unitary in unitary_arr:
            for class1, class2 in class_pairs:
                for resizing_method in resize:
                    start_time = time.time()
                    print(f"Parameters: dataset {dataset}, classs {class1} and {class2}, unitary {unitary}, embedding {embedding}, resize {resizing_method} ")
                    x_train, x_test, y_train, y_test = data.data_load_and_process(class1, class2,resizing_method,dataset)
                
                    trained_params = training.circuit_training(x_train, y_train, unitary, embedding)
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    print("Testing...")
                    predictions = [circuit.QCNN(x, trained_params, unitary, embedding) for x in x_test]
                    test_accuracy = testing.accuracy_test(predictions, y_test)
                    
                    # Escribir los resultados en el archivo
                    f.write(f"Parameters: dataset {dataset}, class {class1} and {class2}, unitary {unitary}, embedding {embedding}, resize {resizing_method}\n")
                    f.write(f"Unitary: {unitary}\n")
                    f.write(f"Training Time: {elapsed_time:.2f} seconds\n")
                    f.write(f"Test accuracy: {test_accuracy * 100:.2f}%\n\n")
                    
                    print(f"classs: {class1} and {class2}")
                    print(f"Time Training: {elapsed_time:.2f} seconds")
                    print(f"Test accuracy: {test_accuracy * 100:.2f}%\n")

    print("Results saved to", results_file)