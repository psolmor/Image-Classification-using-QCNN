import data
import training
import testing
import circuit
import time
import utils

def benchmark(class_pairs, unitary_arr, embedding,resize,dataset):
    results_file = "results.txt"

    # Abrir el archivo en modo de adición
    with open(results_file, "a") as f:
        f.write("#################################\n")
        f.write(f"{time.ctime()}\n")
        for unitary in unitary_arr:
            for class1, class2 in class_pairs:
                for resizing_method in resize:
                    
                    print(f"Parameters: {utils.get_label(class1,dataset)} {utils.get_label(class2,dataset)}, unitary {unitary}, embedding {embedding}, resize {resizing_method} ")
                    x_train, x_test, y_train, y_test = data.data_load_and_process(class1, class2, resizing_method, dataset)
                
                    start_time = time.time()
                    trained_params = training.circuit_training(x_train, y_train, unitary, embedding)
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    print("Testing...")
                    predictions = [circuit.QCNN(x, trained_params, unitary, embedding) for x in x_test]
                    test_accuracy = testing.accuracy_test(predictions, y_test)
                    
                    f.write(f"Parameters:\n{utils.get_label(class1,dataset)} {utils.get_label(class2,dataset)}\n")
                    f.write(f"Unitary: {unitary}\n")
                    f.write(f"Number_parameters: {utils.param_num(unitary)}\n")
                    f.write(f"Embedding: {embedding}\n")
                    f.write(f"Resize: {resizing_method}\n")
                    f.write(f"Training Time: {elapsed_time:.2f} seconds\n")
                    f.write(f"Test accuracy: {test_accuracy * 100:.2f}%\n\n")
                    
                    print(f"Time Training: {elapsed_time:.2f} seconds")
                    print(f"Test accuracy: {test_accuracy * 100:.2f}%\n")

    print("Results saved to", results_file)