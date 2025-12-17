# Recurrent Neural Network from Scratch (RNN)

import numpy as np

# Input sequence and target sequence
input_sequence = [1, 2, 3, 4]
target_sequence = [2, 3, 4, 5]

# Initialize weights
np.random.seed(3)
Wx = np.random.randn(1, 1)   # input weight
Wh = np.random.randn(1, 1)   # hidden state weight
Wy = np.random.randn(1, 1)   # output weight

hidden_state = np.zeros((1, 1))
learning_rate = 0.01
epochs = 1000

# Training loop
for epoch in range(epochs):
    total_loss = 0

    for x_val, y_val in zip(input_sequence, target_sequence):
        x_val = np.array([[x_val]])
        y_val = np.array([[y_val]])

        # Forward pass
        hidden_state = np.tanh(np.dot(Wx, x_val) + np.dot(Wh, hidden_state))
        predicted = np.dot(Wy, hidden_state)

        # Error calculation
        error = predicted - y_val
        total_loss += error ** 2

        # Backpropagation (simplified)
        dWy = 2 * error * hidden_state.T
        dHidden = 2 * error * Wy * (1 - hidden_state ** 2)
        dWh = dHidden * hidden_state.T
        dWx = dHidden * x_val.T

        # Update weights
        Wy = Wy - learning_rate * dWy
        Wh = Wh - learning_rate * dWh
        Wx = Wx - learning_rate * dWx

    if epoch % 200 == 0:
        print("Epoch:", epoch, "Loss:", total_loss)

print("RNN training completed")
