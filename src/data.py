import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 


def data_load_and_process():
    
    print("Loading data...")
    (x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()


    print("Filtering data...")
    x_train_filtered = []
    y_train_filtered = []
    for i in range(len(y_train)):
        if y_train[i] == 0 or y_train[i] == 1:
            x_train_filtered.append(x_train[i])
            y_train_filtered.append(y_train[i])

    x_test_filtered = []
    y_test_filtered = []
    for i in range(len(y_test)):
        if y_test[i] == 0 or y_test[i] == 1:
            x_test_filtered.append(x_test[i])
            y_test_filtered.append(y_test[i])

    x_train = np.array(x_train_filtered)
    y_train = np.array(y_train_filtered)
    x_test = np.array(x_test_filtered)
    y_test = np.array(y_test_filtered)

    """"
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for i, ax in enumerate(axes):
        ax.imshow(x_train[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"Label: {y_train[i]}")
        ax.axis('off')
    plt.show()
    """

    print("Data loaded, normalization...")    

    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  



    print("Resizing image with linear interpolation and flattening vector...")

    x_train = tf.image.resize(x_train[:], (256, 1), method="bilinear").numpy().squeeze()
    x_test = tf.image.resize(x_test[:], (256, 1), method="bilinear").numpy().squeeze()

    print("Data is ready")

    return x_train, x_test, y_train, y_test

