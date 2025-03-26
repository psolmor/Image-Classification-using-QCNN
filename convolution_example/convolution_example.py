import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

def plot_comparison(original, resized, title="Comparison of Images"):
    num_images = len(original)
    fig, axes = plt.subplots(2, num_images, figsize=(8, 4 * num_images))

    for i in range(num_images):
        # Mostrar imagen original
        axes[0, i].imshow(original[i], cmap="gray")
        axes[0, i].set_title("Original (28x28)")
        axes[0, i].axis("off")

        # Mostrar imagen redimensionada
        axes[1, i].imshow(resized[i], cmap="gray")
        axes[1, i].set_title("Resized (16x16)")
        axes[1, i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("Original vs Resized")
    plt.show()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_sample = np.array([x_train[i] for i in range(3)])

x_train_sample_normalized = x_train_sample / 255.0


x_train_sample_expanded = np.expand_dims(x_train_sample_normalized, axis=-1)


x_train_sample_resized = tf.image.resize(x_train_sample_expanded, (16, 16), method="bilinear").numpy()


x_train_sample_resized = np.squeeze(x_train_sample_resized, axis=-1)


np.save("resized_images.npy", x_train_sample_resized)


plot_comparison(x_train_sample, x_train_sample_resized, "Original vs Resized")

print("Imágenes redimensionadas guardadas como 'resized_images.npy'")