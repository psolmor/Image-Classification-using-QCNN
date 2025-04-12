import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 


def plot_comparison(original, bilinear, nearest, area, title="Comparison of Images"):
    num_images = len(original)
    fig, axes = plt.subplots(num_images, 4, figsize=(12, 3 * num_images))  

    for i in range(num_images):
        # Mostrar imagen original
        axes[i, 0].imshow(original[i], cmap="gray")
        axes[i, 0].set_title("Original (28x28)", fontsize=10)
        axes[i, 0].axis("off")

        # Mostrar imagen redimensionada con interpolación bilineal
        axes[i, 1].imshow(bilinear[i], cmap="gray")
        axes[i, 1].set_title("Bilinear (16x16)", fontsize=10)
        axes[i, 1].axis("off")

        # Mostrar imagen redimensionada con interpolación nearest
        axes[i, 2].imshow(nearest[i], cmap="gray")
        axes[i, 2].set_title("Nearest (16x16)", fontsize=10)
        axes[i, 2].axis("off")

        # Mostrar imagen redimensionada con interpolación area
        axes[i, 3].imshow(area[i], cmap="gray")
        axes[i, 3].set_title("Area (16x16)", fontsize=10)
        axes[i, 3].axis("off")

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

x_train_sample_area_expanded = np.expand_dims(x_train_sample_normalized, axis=-1)
x_train_sample_area_resized = tf.image.resize(x_train_sample_area_expanded, (16, 16), method="area").numpy()
x_train_sample_area_resized = np.squeeze(x_train_sample_area_resized, axis=-1)


plot_comparison(x_train_sample, x_train_sample_bil_resized,x_train_sample_near_resized,x_train_sample_area_resized, "Original vs Resized")

