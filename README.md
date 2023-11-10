# Introduction to Deep Learning Library
This library provides foundational building blocks for creating and training simple neural networks. It was developed as study material for the "Introduction to Deep Learning" course at PACE University, 2023.


## Features
- Layer Base Class: A foundational class for creating custom layers.
- Fully Connected Layer (FCLayer): Implements a fully connected neural network layer.
- Activation Layers: Supports tanh, Softmax, and ReLU activation functions.
- Loss Functions: Implements the mean squared error (mse) and binary cross-entropy.
- Pooling Layer: Implements max pooling.
- Convolutional Layer: Implements 2D convolution.
- Flatten Layer: Flattens the input.
- Network Class: Represents a neural network. It allows adding layers, training the network, and making predictions.

## Usage
### Creating a Network:
```python
net = Network()
```
### Adding Layers:
```python
net.add(FlattenLayer())
net.add(FCLayer(input_size, output_size))
net.add(ActivationLayer(activation_function='tanh'))
```
### Training:
```python 
net.fit(x_train, y_train, epochs = 20, learning_rate = 0.001, evaluation=0.2)
```

### Prediction
```python 
predictions = net.predict(input_data)
```
### Evaluation
```python 
accuracy = net.evaluation(x_test, y_test)
```

### MNIST Example
```python
from ann import Network
from layers import FlattenLayer, ActivationLayer, FCLayer
from keras.datasets import mnist
from keras import utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 1, 28*28)
x_test = x_test.reshape(10000, 1, 28*28)

x_train = x_train/255
x_test = x_test/255

y_train = utils.to_categorical(y_train)

model = Network()
model.add(FlattenLayer())
model.add(FCLayer(28*28,10))
model.add(ActivationLayer())

model.fit(x_train,y_train, 20, 0.01, evaluation = 0.1)

```

## Dependencies
- numpy
- scipy
  
## Installation
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
```bash
git clone https://github.com/Bakhitovd/IntroDL
```
