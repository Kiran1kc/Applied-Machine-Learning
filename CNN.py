# Convolutional Neural Network from Scratch- 1D Convolution

import numpy as np

# Input
input_signal = np.array([1, 2, 3, 4, 5])

# Filter
kernel = np.array([1, -1])

# operation
convolution_result = []

for i in range(len(input_signal) - len(kernel) + 1):
    current_region = input_signal[i:i + len(kernel)]
    conv_value = np.sum(current_region * kernel)
    convolution_result.append(conv_value)

convolution_result = np.array(convolution_result)

# Activation function- ReLU
relu_output = []
for value in convolution_result:
    if value > 0:
        relu_output.append(value)
    else:
        relu_output.append(0)

relu_output = np.array(relu_output)

print("Convolution Output:")
print(convolution_result)
print("After ReLU Activation:")
print(relu_output)
