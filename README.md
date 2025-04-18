# Quantum Convolutional Neural Network (QCNN) for Image Classification

<img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python"> <img src="https://img.shields.io/badge/Framework-PennyLane-blueviolet" alt="PennyLane"> <img src="https://img.shields.io/badge/Quantum-QCNN-success" alt="Quantum">

Development of a quantum convolutional neural network for image classification. This code specializes in binary classification. Different convolution and embedding methods will be used for testing.  

Use `python main.py` to run the benchmark.

## Components
1. `data.py` 
   - Prepares a dataset. Currently, it transforms the 28x28 px image to a 16x16 px image in order to fit it into 8 qubits. Therefore, it returns the data in 1-D arrays.
   - The function `data_load_and_process` loads the dataset, normalizes the images, resizes them, and adjusts the labels for binary classification.
   - The dataset is filtered in the way of having just two possible classes in the dataset, this is done to simplify the model.

2. `embedding.py`
   - Contains functions for embedding classical data into quantum states.
   - The `data_embedding` function takes the preprocessed data and embeds it into a quantum state using amplitude encoding. 

3. `QCNN_circuit.py`
   - Defines the quantum convolutional neural network (QCNN) circuit. This file contains the overall architecture of the QCNN.
   - The `qcnn` function constructs the QCNN circuit and returns the output probabilities.

5. `components.py`
   - Defines  unitary operations and layers used in the QCNN. .

6. `training.py`
   - Contains the training loop for the QCNN. This file includes functions to train the QCNN using a specified dataset. Corrently used "mse"
   - The `circuit_training` function trains the QCNN using the Adam optimizer and returns the loss history and trained parameters.

7. `utils.py`
   - Contains smaller functions used in the code.

## References

1. **Quantum Convolutional Neural Networks**  
   Henderson et al.  
   [arXiv:2108.00661](https://arxiv.org/abs/2108.00661)  
   *Original paper inspiring this implementation*

2. **PennyLane Quantum Machine Learning**  
   Xanadu AI  
   [PennyLane Documentation](https://pennylane.ai/)  
   *Quantum framework used for circuit construction*

3. **MNIST Dataset**  
   Yann LeCun et al.  
   [Official MNIST Website](http://yann.lecun.com/exdb/mnist/)  
   *Standard handwritten digit dataset*

4. **Fashion-MNIST Dataset**  
   Zalando Research  
   [GitHub Repository](https://github.com/zalandoresearch/fashion-mnist)  
   *Clothing item classification dataset*

5.  **Quantum Computation and Quantum Information**  
   Michael A. Nielsen & Isaac L. Chuang  
   Cambridge University Press (2010)  
   *The standard textbook on quantum computing fundamentals*  
   [ISBN: 978-1-107-00217-3](https://doi.org/10.1017/CBO9780511976667)

