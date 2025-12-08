import numpy as np

### 1. Core Functions (Activation and Loss)
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

def sigmoid_backward(dA, Z):
    S = 1 / (1 + np.exp(-Z))
    dZ = dA * S * (1 - S)
    return dZ

def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def compute_cost(A_L, Y):
    m = Y.shape[1]
    # Cross-Entropy Loss
    cost = (-1 / m) * np.sum(Y * np.log(A_L) + (1 - Y) * np.log(1 - A_L))
    cost = np.squeeze(cost)
    return cost

### 2. Initialization and Layer Ops
def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache

### 3. Backpropagation
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters

### 4. Model (L-Layer implementation)
def ann_model(X, Y, layer_dims, learning_rate=0.01, num_iterations=1000):
    parameters = initialize_parameters(layer_dims)
    costs = []
    L = len(layer_dims) - 1 # number of layers
    
    for i in range(0, num_iterations):
        # --- Forward Propagation ---
        A = X
        caches = []
        for l in range(1, L): # Hidden Layers (ReLU)
            A_prev = A
            A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
            caches.append(cache)
        
        # Output Layer (Sigmoid)
        AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
        caches.append(cache)
        
        # --- Compute Cost ---
        cost = compute_cost(AL, Y)
        
        # --- Backward Propagation ---
        grads = {}
        # Initializing backpropagation (from Cost)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        # Output Layer (Sigmoid)
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
        
        # Hidden Layers (ReLU)
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            
        # --- Update Parameters ---
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 100 == 0:
            costs.append(cost)
            # print (f"Cost after iteration {i}: {cost}")
            
    return parameters, costs

# Example usage (Dummy Data)
# X: 2 features, 100 examples. Y: 1 output, 100 examples.
# Note: X and Y are transposed for the standard ML formulation.
X_train = np.random.randn(2, 100) 
Y_train = (np.sum(X_train, axis=0, keepdims=True) > 0).astype(int) 

layer_dimensions = [2, 4, 1] # Input=2, Hidden=4, Output=1
# final_parameters, costs = ann_model(X_train, Y_train, layer_dimensions, num_iterations=2000)
# print("Training Complete. Final Cost:", costs[-1])