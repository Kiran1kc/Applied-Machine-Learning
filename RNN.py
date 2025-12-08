### 1. Initialization
def initialize_rnn_parameters(n_a, n_x, n_y):
    """
    n_a -- dimension of the hidden state
    n_x -- dimension of the input X
    n_y -- dimension of the output Y
    """
    parameters = {}
    parameters['Waa'] = np.random.randn(n_a, n_a) * 0.01
    parameters['Wax'] = np.random.randn(n_a, n_x) * 0.01
    parameters['Wya'] = np.random.randn(n_y, n_a) * 0.01
    parameters['ba'] = np.zeros((n_a, 1))
    parameters['by'] = np.zeros((n_y, 1))
    return parameters

### 2. RNN Cell Forward (Single Time Step)
def rnn_cell_forward(xt, a_prev, parameters):
    """
    xt -- Input at time step t (n_x, m)
    a_prev -- Hidden state at time step t-1 (n_a, m)
    """
    Waa, Wax, Wya, ba, by = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['ba'], parameters['by']

    # Hidden State Update (tanh activation)
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    
    # Output Prediction (softmax for classification, or other)
    yt_pred = np.dot(Wya, a_next) + by
    
    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache

### 3. RNN Forward Pass (Unrolled)
def rnn_forward(x, a0, parameters):
    """
    x -- Input data for all time steps (n_x, m, T_x)
    a0 -- Initial hidden state (n_a, m)
    """
    (n_x, m, T_x) = x.shape
    n_a = parameters['Waa'].shape[0]
    n_y = parameters['Wya'].shape[0]
    
    # Initialize a and y_pred
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    
    # Initialize a_next (a_prev for the loop)
    a_next = a0
    caches = []
    
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
        
    return a, y_pred, caches

### 4. BPTT (RNN Cell Backward) - Summing Gradients Over Time
def rnn_cell_backward(da_next, cache):
    """
    da_next -- Gradient of cost with respect to a_next (n_a, m)
    """
    (a_next, a_prev, xt, parameters) = cache
    Waa, Wax, Wya, ba, by = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['ba'], parameters['by']
    
    # Use d(tanh(u))/du = 1 - tanh^2(u)
    # The '1 - a_next**2' is the derivative of tanh(Z) *with respect to Z*
    dtanh = (1 - a_next**2) * da_next
    
    # Gradients of parameters for this time step
    dWaa = np.dot(dtanh, a_prev.T)
    dWax = np.dot(dtanh, xt.T)
    dba = np.sum(dtanh, axis=1, keepdims=True)
    
    # Gradients for previous hidden state and input
    da_prev = np.dot(Waa.T, dtanh)
    dxt = np.dot(Wax.T, dtanh)
    
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradients

# NOTE: The full RNN backward function involves looping over time and summing the gradients (dWaa, dWax, dba) calculated at each step to get the final total gradient. The calculation for dWya and dby is derived from the output loss and uses the chain rule with the Wya/by terms.