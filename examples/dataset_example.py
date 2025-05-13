import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def plot_examples(dataset_name, images, labels, class_names, num_examples=10):
    """
    Función para graficar ejemplos de un conjunto de datos.
    
    Args:
        dataset_name (str): Nombre del conjunto de datos (e.g., "Fashion MNIST", "MNIST").
        images (numpy.ndarray): Imágenes del conjunto de datos.
        labels (numpy.ndarray): Etiquetas correspondientes a las imágenes.
        class_names (list): Lista de nombres de las clases.
        num_examples (int): Número de ejemplos a graficar.
    """
    plt.figure(figsize=(10, 5))
    indices = np.random.choice(len(images), num_examples, replace=False)  # Seleccionar imágenes aleatorias
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)  # Crear una cuadrícula de 2 filas y 5 columnas
        plt.imshow(images[idx], cmap="gray")
        plt.title(f"{class_names[labels[idx]]}", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Cargar Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train_fashion, y_train_fashion), (_, _) = fashion_mnist.load_data()
fashion_class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Cargar MNIST
mnist = tf.keras.datasets.mnist
(x_train_mnist, y_train_mnist), (_, _) = mnist.load_data()
mnist_class_names = [str(i) for i in range(10)]  # Clases del 0 al 9

# Graficar ejemplos de Fashion MNIST
plot_examples("Fashion MNIST", x_train_fashion, y_train_fashion, fashion_class_names, num_examples=10)

# Graficar ejemplos de MNIST
plot_examples("MNIST", x_train_mnist, y_train_mnist, mnist_class_names, num_examples=10)