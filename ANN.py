# Artificial Neural Network from Scratch (ANN)
import numpy as np

# Toy dataset (AND logic)
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

expected_output = np.array([[0], [0], [0], [1]])

# Initializing weights and bias randomly
np.random.seed(2)
weights = np.random.randn(2, 1)
bias = np.zeros((1,))

learning_rate = 0.1
epochs = 5000

# Sigmoid activation function
def sigmoid(value):
    return 1 / (1 + np.exp(-value))

# Training the ANN
for epoch in range(epochs):
    # Forward pass
    net_input = np.dot(inputs, weights) + bias
    predicted_output = sigmoid(net_input)

    # Mean Squared Error loss
    error = expected_output - predicted_output
    loss = np.mean(error ** 2)

    # Backpropagation
    gradient = error * predicted_output * (1 - predicted_output)
    weight_gradient = np.dot(inputs.T, gradient)
    bias_gradient = np.sum(gradient)

    # Updating parameters
    weights = weights + learning_rate * weight_gradient
    bias = bias + learning_rate * bias_gradient

# Final prediction
print("Final ANN Output:")
print(np.round(predicted_output))
