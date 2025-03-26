import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

def plot_images(image_array):
    figures, axis = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j  
            axis[i, j].imshow(image_array[idx], cmap="gray") 
            axis[i, j].axis("off") 
    plt.show()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train_sample = [x_train[i] for i in range(9)]
plot_images(x_train_sample)


x_train = x_train[..., np.newaxis] / 255.0
x_train = tf.image.resize(x_train[:], (256, 1), method="bilinear").numpy().squeeze()

