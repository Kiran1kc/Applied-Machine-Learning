# Artificial Neural Network from Scratch (ANN)
import numpy as np

# Toy dataset: XOR problem
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Network parameters
input_size = 2
hidden_size = 2
output_size = 1
lr = 0.1
epochs = 10000

# Initialize weights
np.random.seed(42)
W1 = np.random.rand(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.rand(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    # Loss (MSE)
    loss = np.mean((y - a2)**2)
    
    # Backpropagation
    d_a2 = (a2 - y) * sigmoid_derivative(a2)
    d_W2 = np.dot(a1.T, d_a2)
    d_b2 = np.sum(d_a2, axis=0, keepdims=True)
    
    d_a1 = np.dot(d_a2, W2.T) * sigmoid_derivative(a1)
    d_W1 = np.dot(X.T, d_a1)
    d_b1 = np.sum(d_a1, axis=0, keepdims=True)
    
    # Update weights
    W2 -= lr * d_W2
    b2 -= lr * d_b2
    W1 -= lr * d_W1
    b1 -= lr * d_b1

print("Predictions after training:")
print(a2)
