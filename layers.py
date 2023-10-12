# Studying materials for the "Introduction to Deep Learning" course
# author: Dmitrii Bakhitov
# PACE University 2023

import numpy as np
from scipy.signal import correlate2d, convolve2d

# tanh activation function
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
    
    def number_parameters(self):
        return 0

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    
    def number_parameters(self):
        return self.weights.size + self.bias.size
    
    
# Activation layer class
class ActivationLayer(Layer):
    def __init__(self, activation_function = 'tanh'):
        if activation_function == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


class Softmax(Layer):
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))  # subtract max to stabilize the computation
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.softmax(input_data)
        return self.output

    def backward_propagation(self, output_error, learning_rate):

        softmax_gradient = self.output * (1 - self.output)
        return output_error * softmax_gradient

class ReLU(Layer):

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_prime(self, x):
        """Derivative of the ReLU function."""
        return np.where(x > 0, 1, 0)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.relu(input_data)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return output_error * self.relu_prime(self.input)
    
    
class FlattenLayer(Layer):
    def forward_propagation(self, input_data):
        self.input_shape = input_data.shape
        return input_data.reshape(1, -1)

    def backward_propagation(self, output_error, learning_rate):
        return output_error.reshape(self.input_shape)

    
class MaxPoolingLayer(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None

    def forward_propagation(self, input_data):
        
        # Assuming input_data has shape (num_filters, height, width)
        if len(input_data.shape)==3:
            num_filters, input_height, input_width = input_data.shape
        else:
            input_data = np.array([input_data])
            num_filters, input_height, input_width = input_data.shape
            
        self.input = input_data
        # Calculate output dimensions after the pooling operation
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1

        # Initialize output with zeros
        output = np.zeros((num_filters, output_height, output_width))

        for i in range(num_filters):
            for y in range(0, output_height):
                for x in range(0, output_width):
                    start_y = y * self.stride
                    end_y = min(start_y + self.pool_size, input_height)
                    start_x = x * self.stride
                    end_x = min(start_x + self.pool_size, input_width)
                    
                    output[i, y, x] = np.max(input_data[i, start_y:end_y, start_x:end_x])
        output = output[0] if num_filters == 1 else output
        return output

    def backward_propagation(self, output_error, learning_rate):
        # Assuming output_error has shape (num_filters, height, width)
        if len(output_error.shape)==3:
            num_filters, output_height, output_width = output_error.shape
        else:
            output_error = np.array([output_error])
            num_filters, output_height, output_width = output_error.shape

        d_input = np.zeros(self.input.shape)

        for i in range(num_filters):
            for y in range(0, output_height):
                for x in range(0, output_width):
                    start_y = y * self.stride
                    end_y = min(start_y + self.pool_size, self.input.shape[1])
                    start_x = x * self.stride
                    end_x = min(start_x + self.pool_size, self.input.shape[2])
                    
                    # Find the index of the max value in the input within the pooling window
                    (a, b) = np.unravel_index(
                        np.argmax(self.input[i, start_y:end_y, start_x:end_x]),
                        (end_y - start_y, end_x - start_x)
                    )
                    d_input[i, start_y + a, start_x + b] = output_error[i, y, x]
        d_input = d_input[0] if num_filters == 1 else d_input
        return d_input

    
class ConvLayer(Layer):
    def __init__(self ,num_filters, kernel_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mode = 'valid' if padding == 0 else 'same'
        # Initialize filters and bias with small random values
        self.filters = np.random.randn(num_filters, kernel_size, kernel_size) * 0.01
        self.bias = np.random.randn(num_filters, 1) * 0.01


    def forward_propagation(self, input_data):
        self.input = input_data
        input_height, input_width = input_data.shape
        output_height = (input_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (input_width - self.kernel_size + 2 * self.padding) // self.stride + 1

        # Initialize output with zeros
        self.output = np.zeros((self.num_filters, output_height, output_width))
        for f in range(self.num_filters):
            self.output[f] = correlate2d(input_data, self.filters[f], mode=self.mode)
            self.output[f] += self.bias[f]

        return self.output




    def backward_propagation(self, output_error, learning_rate):

        d_filters = np.zeros(self.filters.shape)
        d_bias = np.sum(output_error, axis=(1, 2)).reshape(self.bias.shape)
        d_input = np.zeros(self.input.shape)

        for f in range(self.num_filters):
            
            d_filters[f] = correlate2d(self.input, output_error[f], "valid")
            d_input += convolve2d(output_error[f], self.filters[f], "full")
    
        # Update filters and bias
        self.filters -= learning_rate * d_filters
        self.bias -= learning_rate * d_bias

        return d_input
