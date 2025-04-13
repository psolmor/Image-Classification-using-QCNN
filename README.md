# Image-Classification-using-QCNN
Development of a quantum convolutional neural network for image classification. This code specializes in binary classification. Different convolution and embedding methods will be used for testing.  

The whole project is written in python. Tensorflow and numpy is required for the data treatment. The quantum simulation is done using the framework [pennylane](https://pennylane.ai/)

This work takes direct inspiration from this [previous work](https://arxiv.org/abs/2108.00661)

Use `main.py` train and test de QCNN.

## Components
1. `data.py` 
   - Prepares a dataset from NIST. Currently, it transforms the 28x28 px image to a 16x16 px image in order to fit it into 8 qubits. Therefore, it returns the data in 1-D arrays.
   - The function `data_load_and_process` loads the dataset, normalizes the images, resizes them, and adjusts the labels for binary classification.
   - The dataset is filtered in the way of having just two possible classes in the dataset, this is done to simplify the model.

2. `embedding.py`
   - Contains functions for embedding classical data into quantum states.
   - The `data_embedding` function takes the preprocessed data and embeds it into a quantum state using amplitude encoding. By default the chosen method will be amplitude embedding for 8 qubits.

3. `QCNN_circuit.py`
   - Defines the quantum convolutional neural network (QCNN) circuit. This file contains the overall architecture of the QCNN.
   - The `qcnn` function constructs the QCNN circuit and returns the output probabilities.

5. `components.py`
   - Defines  unitary operations and alyers used in the QCNN. .

6. `training.py`
   - Contains the training loop for the QCNN. This file includes functions to train the QCNN using a specified dataset. Corrently used "mse"
   - The `circuit_training` function trains the QCNN using the Adam optimizer and returns the loss history and trained parameters.

7. `utils.py`
   - Contains smaller functions used in the code.
