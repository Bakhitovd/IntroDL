# Studying materials for the "Introduction to Deep Learning" course
# author: Dmitrii Bakhitov
# PACE University 2023

import numpy as np

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

    
    
#     tanh activation function
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;


# mean squared error 
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size  


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

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
    
     
    
# the model class    
class Network:
    def __init__(self):
        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    #def use(self, loss, loss_prime):
    #    self.loss = loss
    #    self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagati on
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result
    
    def evaluation(self, x, y):
        """
        
        x - test data
        y - test labels
        returns the classification accuracy  
        
        """
        prediction = self.predict(x)
        acc = 0
        for i in range(len(y)):
            pred = np.argmax(prediction[i][0])
            if pred == y[i]:
                acc += 1

        return acc/len(y)

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate, evaluation = 0):
        """
        x_train - training dataset
        y_train - training labels
        epochs - number of epochs to train
        learning_rate - learning rate, Ex: learning_rate = 0.001
        evaluation - portion of data to use for evaluation, Ex: 0.25 - 25% of data is used for validation
        """
        # create variables to store trainning results 
        self.training = {}
        epochs_list = []
        err_list = []
        eval_err_list = []
        eval_acc_list = []
        
        # sample dimension first
        samples = len(x_train)
        eval_samples = 0
        # split training in case evaluation > 0
        if evaluation != 0:
            
            eval_samples = int(evaluation*samples)
            samples = samples - eval_samples
            self.eval_samples = eval_samples
            self.samples = samples
            
            x_eval = x_train[samples:]
            x_train = x_train[:samples]
            y_eval = y_train[samples:]
            y_train = y_train[:samples]

        # training loop
        for i in range(epochs):
            err = 0
            eval_err = 0
            eval_acc = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            
            err /= samples
            epochs_list.append(i+1)
            err_list.append(err)
            
            
            # evaluation step
            if evaluation != 0:
                for j in range(eval_samples):
                    output = x_eval[j]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)
                        
                    # compute loss (for display purpose only)
                    eval_err += self.loss(y_eval[j], output)
                    
                eval_acc = self.evaluation(x_eval, np.argmax(y_eval,axis=1))
                eval_err /= eval_samples
            
            eval_err_list.append(eval_err)
            eval_acc_list.append(eval_acc)
            
            print('epoch %d/%d | training_loss=%f | eval_loss=%f | eval_accuracy=%f' % (i+1, epochs, err, eval_err, eval_acc))
        self.training = {'epoch':epochs_list, 'training_loss': err_list,'eval_loss':eval_err_list, 'eval_accuracy': eval_acc_list}