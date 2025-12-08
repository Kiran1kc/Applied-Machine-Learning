### 1. Utility Functions
def zero_pad(X, pad):
    # Pads X with zeros around the border
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    # Applies one filter to a single slice of the input
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + float(b)
    return Z

### 2. Forward Propagation
def conv_forward(A_prev, W, b, hparameters):
    """
    Arguments:
    A_prev -- Output activations of the previous layer (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights/Filters (f, f, n_C_prev, n_C)
    b -- Biases (1, 1, 1, n_C)
    hparameters -- Python dictionary containing "stride" and "pad"
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev_W, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Calculate output dimensions
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    # Initialize output volume Z
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Apply padding
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]   # (n_H_prev_pad, n_W_prev_pad, n_C_prev)
        for h in range(n_H):           # loop over vertical axis of the output
            for w in range(n_W):       # loop over horizontal axis of the output
                for c in range(n_C):   # loop over the filters (channels of output)

                    # Find the corners of the current slice
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Slice the previous activation data
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Convolve the slice with the filter W and bias b
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

    cache = (A_prev, W, b, hparameters)
    return Z, cache

### 3. Backward Propagation (Simplified for Clarity)
def conv_backward(dZ, cache):
    """
    This function implements the three main gradients: dA_prev, dW, db.
    """
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev_W, n_C) = W.shape
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize gradients
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    
                    # Find the corners of the current slice
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Slice (A_prev) and Gradient Slice (dZ)
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    dz = dZ[i, h, w, c]
                    
                    # Update Gradients (dW, db, dA_prev)
                    # dW calculation (Convolution of input slice with dZ)
                    dW[:, :, :, c] += a_slice * dz
                    # db calculation (Sum of dZ over all positions and examples)
                    db[:, :, :, c] += dz
                    # dA_prev calculation (Convolution of W with dZ - requires index matching)
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dz

        # Set the unpadded part of the dA_prev (after removing the border)
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
            
    return dA_prev, dW, db

# Example usage (Dummy Data)
# A_prev: 1 image, 4x4, 3 channels. W: 2x2 filter, 3 input channels, 2 output filters.
# A_prev_test = np.random.randn(1, 4, 4, 3) 
# W_test = np.random.randn(2, 2, 3, 2)
# b_test = np.random.randn(1, 1, 1, 2)
# hparameters = {"pad": 1, "stride": 2}
# Z, cache_conv = conv_forward(A_prev_test, W_test, b_test, hparameters)
# print("Z shape:", Z.shape)