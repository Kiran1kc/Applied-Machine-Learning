import numpy as np

# Toy dataset: sequence sum modulo 2
X = np.array([[0,1,0,1],[1,0,1,0],[1,1,0,0]])
y = np.array([[0],[0],[1]])

# RNN parameters
input_size = 1
hidden_size = 2
output_size = 1
lr = 0.1
epochs = 5000

# Initialize weights
np.random.seed(42)
Wx = np.random.rand(input_size, hidden_size)
Wh = np.random.rand(hidden_size, hidden_size)
Wy = np.random.rand(hidden_size, output_size)
bh = np.zeros((1, hidden_size))
by = np.zeros((1, output_size))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

# Training
for epoch in range(epochs):
    total_loss = 0
    for xi, yi in zip(X, y):
        xi = xi.reshape(-1,1)
        h = np.zeros((xi.shape[0]+1, hidden_size))
        
        # Forward pass
        for t in range(xi.shape[0]):
            h[t+1] = sigmoid(np.dot(xi[t], Wx) + np.dot(h[t], Wh) + bh)
        y_pred = sigmoid(np.dot(h[-1], Wy) + by)
        
        loss = (y_pred - yi)**2
        total_loss += loss
        
        # Backprop through time (simple, 1 step)
        d_y = (y_pred - yi) * sigmoid_derivative(y_pred)
        d_Wy = np.dot(h[-1].reshape(-1,1), d_y.reshape(1,-1))
        d_by = d_y
        
        d_h = np.dot(d_y, Wy.T) * sigmoid_derivative(h[-1])
        d_Wx = np.dot(xi[-1].reshape(-1,1), d_h.reshape(1,-1))
        d_Wh = np.dot(h[-2].reshape(-1,1), d_h.reshape(1,-1))
        d_bh = d_h
        
        # Update
        Wy -= lr * d_Wy
        by -= lr * d_by
        Wx -= lr * d_Wx
        Wh -= lr * d_Wh
        bh -= lr * d_bh
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss}")

print("Predictions after training:")
for xi in X:
    xi = xi.reshape(-1,1)
    h = np.zeros((xi.shape[0]+1, hidden_size))
    for t in range(xi.shape[0]):
        h[t+1] = sigmoid(np.dot(xi[t], Wx) + np.dot(h[t], Wh) + bh)
    y_pred = sigmoid(np.dot(h[-1], Wy) + by)
    print(y_pred)
