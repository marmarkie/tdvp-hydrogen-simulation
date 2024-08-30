"""
    This code expands the soft Coulomb potential into 5 Gaussians to facilitate the analytical calculations of the matrix elements. Specifically,

        V(r) = -1 / sqrt(r^2 + a^2) = Σ Ci * exp(-σi * r^2), (0 <= i < 5)

    The parameters Ci and σi are obtained using optimizer in tensorflow.
    
    The results are stored in `V0_expansion_params_tf.npz` with keys "values_C" and "values_σ".
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, initializers
import os

# Change to your path
os.chdir('E:\\XUTTAI\\Desktop\\tdvp\\potential_expansion')

# Custom Gaussian activation layer e^(-b^2 * r^2), where b is a trainable parameter
class GaussianLayer(layers.Layer):
    def __init__(self, n_gaussians, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)
        self.n_gaussians = n_gaussians

    def build(self, input_shape):
        # Initialize trainable parameters a and b
        self.a = self.add_weight(shape=(self.n_gaussians,), initializer=initializers.RandomNormal(), trainable=True, name='a')
        self.b = self.add_weight(shape=(self.n_gaussians,), initializer=initializers.RandomNormal(), trainable=True, name='b')

    def call(self, inputs):
        r = inputs
        r_squared = tf.square(r)
        gaussians = -self.a ** 2 * tf.exp(-self.b ** 2 * r_squared)
        return tf.reduce_sum(gaussians, axis=-1, keepdims=True)

# Build the neural network model
def build_model(n_gaussians):
    model = models.Sequential([
        layers.Input(shape=(1,)),
        GaussianLayer(n_gaussians=n_gaussians)
    ])
    return model

# Generate data
def generate_data(a, r_max, num_points):
    r = np.linspace(0, r_max, num_points)
    y = -1 / np.sqrt(r ** 2 + a ** 2)
    return r, y

# Train the model
def train_model(model, r, y, epochs=100, batch_size=32):
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(r, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return history

# Extract trained Gaussian parameters
def extract_gaussian_params(gaussian_layer, n_gaussians):
    c_values = -gaussian_layer.get_weights()[0] ** 2
    beta_values = gaussian_layer.get_weights()[1]

    gaussian_params = {}
    for i in range(n_gaussians):
        gaussian_params[f'Gaussian {i+1}'] = {'C': np.complex128(c_values[i]), 'σ': np.complex128(beta_values[i] ** 2)}

    return gaussian_params

# Save parameters to a .npz file
def save_params(filepath, gaussian_params):
    keys = list(gaussian_params.keys())
    values_C = np.array([gaussian_params[key]['C'] for key in keys], dtype=np.complex128)
    values_σ = np.array([gaussian_params[key]['σ'] for key in keys], dtype=np.complex128)
    np.savez(filepath, keys=keys, values_C=values_C, values_σ=values_σ)

# Plot results
def plot_results(r, y, y_pred, history, n_gaussians):
    # Compare model prediction with actual data
    plt.figure(figsize=(10, 6))
    plt.plot(r, y, label='V(r) = exp(-r)', color='blue', lw=2)
    plt.plot(r, y_pred, label=f'TensorFlow Gaussian Model (n={n_gaussians})', color='red', linestyle='--', lw=2)
    plt.xlabel('r')
    plt.ylabel('Function value')
    plt.title('Comparison between V(r) and TensorFlow Gaussian Model Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the loss function over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='green', lw=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    # Define parameters
    a = 1.0
    r_max = 100
    num_points = 10000
    n_gaussians = 50
    filepath = 'V0_expansion_params_tf.npz'

    # Generate data
    r, y = generate_data(a, r_max, num_points)

    # Build and train the model
    model = build_model(n_gaussians)
    history = train_model(model, r, y, epochs=100, batch_size=32)

    # Predict results
    y_pred = model.predict(r)

    # Extract and save Gaussian parameters
    gaussian_layer = model.layers[0]
    gaussian_params = extract_gaussian_params(gaussian_layer, n_gaussians)
    save_params(filepath, gaussian_params)

    # Plot results
    plot_results(r, y, y_pred, history, n_gaussians)

    # Print final loss value
    print(f"Final loss: {history.history['loss'][-1]}")

if __name__ == '__main__':
    main()
