import numpy as np
import pennylane.numpy as pnp
import tensorflow as tf
from tensorflow.keras import layers, losses, Model

def data_load_and_process(class1, class2, resize, loss, dataset="mnist"):
    print("Loading data....")
    
    if dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_filter = np.isin(y_train, [class1, class2])
    test_filter = np.isin(y_test, [class1, class2])
    x_train, y_train = x_train[train_filter], y_train[train_filter]
    x_test, y_test = x_test[test_filter], y_test[test_filter]
    
    if loss=="mse":
        y_train = np.where(y_train == class1, 1, -1)
        y_test = np.where(y_test == class1, 1, -1)
    else:
        y_train = np.where(y_train == class1, 1, 0)
        y_test = np.where(y_test == class1, 1, 0)

    x_train = x_train[..., np.newaxis] / 255.0
    x_test = x_test[..., np.newaxis] / 255.0

    if resize in ["bilinear", "area", "nearest"]:
        method = {"bilinear": tf.image.ResizeMethod.BILINEAR,
                  "area": tf.image.ResizeMethod.AREA,
                  "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR}[resize]
        
        x_train = tf.image.resize(x_train, (256, 1), method=method).numpy().squeeze()
        x_test = tf.image.resize(x_test, (256, 1), method=method).numpy().squeeze()

    elif resize == "autoencoder":
        latent_dim = 8  # Puedes parametrizarlo si quieres otras variantes

        class SimpleAutoencoder(Model):
            def __init__(self, latent_dim):
                super(SimpleAutoencoder, self).__init__()
                self.encoder = tf.keras.Sequential([
                    layers.Flatten(),
                    layers.Dense(latent_dim, activation='relu'),
                ])
                self.decoder = tf.keras.Sequential([
                    layers.Dense(784, activation='sigmoid'),
                    layers.Reshape((28, 28))
                ])

            def call(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        autoencoder = SimpleAutoencoder(latent_dim)
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

        autoencoder.fit(x_train, x_train,
                        epochs=10,
                        shuffle=True,
                        validation_split=0.1)

        x_train = autoencoder.encoder(x_train).numpy()
        x_test = autoencoder.encoder(x_test).numpy()

        x_train = (x_train - x_train.min()) * (np.pi / (x_train.max() - x_train.min()))
        x_test = (x_test - x_test.min()) * (np.pi / (x_test.max() - x_test.min()))

    print("Data is ready")

    return (pnp.array(x_train), pnp.array(x_test), pnp.array(y_train), pnp.array(y_test))