import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 


def data_load_and_process(class1,class2,resize,dataset="mnist"):
    
    print("Loading data....")
    if dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


    print("Filtering data...")
    x_train_filtered = []
    y_train_filtered = []
    for i in range(len(y_train)):
        if y_train[i] == class1 or y_train[i] == class2:
            x_train_filtered.append(x_train[i])
            y_train_filtered.append(y_train[i])

    x_test_filtered = []
    y_test_filtered = []
    for i in range(len(y_test)):
        if y_test[i] == class1 or y_test[i] == class2:
            x_test_filtered.append(x_test[i])
            y_test_filtered.append(y_test[i])

    x_train = np.array(x_train_filtered)
    y_train = np.array(y_train_filtered)
    x_test = np.array(x_test_filtered)
    y_test = np.array(y_test_filtered)

    y_train = [1 if y==class1 else -1 for y in y_train]
    y_test = [1 if y==class1 else -1 for y in y_test]

    print("Data loaded, normalization...")    

    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  



    print(f"Resizing image using {resize} resizing and flattening vector...")

    if resize=="bilinear":
        x_train = tf.image.resize(x_train[:], (256, 1), method="bilinear").numpy().squeeze()
        x_test = tf.image.resize(x_test[:], (256, 1), method="bilinear").numpy().squeeze()
    elif resize=="area":
        x_train = tf.image.resize(x_train[:], (256, 1), method="area").numpy().squeeze()
        x_test = tf.image.resize(x_test[:], (256, 1), method="area").numpy().squeeze()
    elif resize=="nearest":
        x_train = tf.image.resize(x_train[:], (256, 1), method="nearest").numpy().squeeze()
        x_test = tf.image.resize(x_test[:], (256, 1), method="nearest").numpy().squeeze()

    
    print("Data is ready")

    return x_train, x_test, y_train, y_test

