# Image-Classification-using-QCNN
Development of a quantum convolutional neural network for image classification. This code specializes in binary classification. Different convolution and embedding methods will be used for testing.  

The whole project is written in python. Tensorflow and numpy is required for the data treatment. The quantum simulation is done using the framework [pennylane](https://pennylane.ai/)

This work takes direct inspiration from this [previous work](https://arxiv.org/abs/2108.00661)

## Components
1. `data.py` 
   - Prepares a dataset from NIST. Currently, it transforms the 28x28 px image to a 16x16 px image in order to fit it into 8 qubits. Therefore, it returns the data in 1-D arrays.
   - The function `data_load_and_process` loads the dataset, normalizes the images, resizes them, and adjusts the labels for binary classification.
   - The labels are transformed such that values greater than 4 become 1, and others become 0. This way we have two classes.

2. `embedding.py`
   - Contains functions for embedding classical data into quantum states.
   - The `data_embedding` function takes the preprocessed data and embeds it into a quantum state using amplitude encoding. By default the chosen method will be amplitude embedding for 8 qubits.