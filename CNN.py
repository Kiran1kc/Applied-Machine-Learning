import numpy as np

# Toy dataset: detect [1, 0, 1] in sequences
X = np.array([[1,0,1,0,1,0],
              [0,1,0,1,0,1],
              [1,1,0,1,1,0]])
y = np.array([[1],[0],[1]])

# Hyperparameters
kernel_size = 3
num_filters = 1
lr = 0.1
epochs = 5000

# Initialize kernel and bias
np.random.seed(42)
kernel = np.random.rand(kernel_size)
bias = np.random.rand(1)

# Convolution function (valid padding)
def conv1d(x, k, b):
    return np.array([np.sum(x[i:i+kernel_size]*k)+b for i in range(len(x)-kernel_size+1)])

# Sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for xi, yi in zip(X, y):
        # Forward pass
        conv_out = conv1d(xi, kernel, bias)
        a = sigmoid(np.mean(conv_out))  # simple global average pooling
        loss = (yi - a)**2
        total_loss += loss
        
        # Backprop
        d_a = (a - yi) * sigmoid_derivative(a)
        grad_kernel = d_a * np.ones(kernel_size)/kernel_size * xi[:kernel_size]
        grad_bias = d_a
        
        kernel -= lr * grad_kernel
        bias -= lr * grad_bias
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss}")

print("Predictions after training:")
for xi in X:
    print(sigmoid(np.mean(conv1d(xi, kernel, bias))))
