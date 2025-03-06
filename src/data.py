import numpy as np 
import tensorflow as tf 


#dataset is a string to indicate which data are we using
def data_load_and_process(dataset="mnist"):
    
    print("Loading data...")
    if dataset=="fashion_mnist":
        (x_train,y_train),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()
    elif dataset=="mnist":
        (x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
   
    print("Data loaded, normalization...")    

    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  

    y_train = np.where(y_train > 4, 1, -1)
    y_test = np.where(y_test > 4, 1, -1)

    print("Resizing image with linear interpolation and flattening vector...")

    x_train = tf.image.resize(x_train[:], (256, 1), method="bilinear").numpy().squeeze()
    x_test = tf.image.resize(x_test[:], (256, 1), method="bilinear").numpy().squeeze()

    print("Data is ready")

    return x_train, x_test, y_train, y_test
