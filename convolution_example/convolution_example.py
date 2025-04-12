import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA

def plot_comparison(original, bilinear, nearest, title="Comparison of Images"):
    num_images = len(original)
    fig, axes = plt.subplots(3, num_images, figsize=(8, 4 * num_images))

    for i in range(num_images):
        # Mostrar imagen original
        axes[0, i].imshow(original[i], cmap="gray")
        axes[0, i].set_title("Original (28x28)")
        axes[0, i].axis("off")

        # Mostrar imagen redimensionada con interpolación bilineal
        axes[1, i].imshow(bilinear[i], cmap="gray")
        axes[1, i].set_title("Bilinear Resized (16x16)")
        axes[1, i].axis("off")

        # Mostrar imagen redimensionada con interpolación nearest
        axes[2, i].imshow(nearest[i], cmap="gray")
        axes[2, i].set_title("Nearest Resized (16x16)")
        axes[2, i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("Original_vs_Resized.png")
    plt.show()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_sample = np.array([x_train[i] for i in range(3)])

x_train_sample_normalized = x_train_sample / 255.0


x_train_sample_bil_expanded = np.expand_dims(x_train_sample_normalized, axis=-1)
x_train_sample_bil_resized = tf.image.resize(x_train_sample_bil_expanded, (16, 16), method="bilinear").numpy()
x_train_sample_bil_resized = np.squeeze(x_train_sample_bil_resized, axis=-1)

x_train_sample_near_expanded = np.expand_dims(x_train_sample_normalized, axis=-1)
x_train_sample_near_resized = tf.image.resize(x_train_sample_near_expanded, (16, 16), method="nearest").numpy()
x_train_sample_near_resized = np.squeeze(x_train_sample_near_resized, axis=-1)


plot_comparison(x_train_sample, x_train_sample_bil_resized,x_train_sample_near_resized, "Original vs Resized")

