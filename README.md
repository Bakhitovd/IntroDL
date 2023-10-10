# Introduction to Deep Learning Library
This library provides foundational building blocks for creating and training simple neural networks. It was developed as study material for the "Introduction to Deep Learning" course at PACE University, 2023.


## Features
- Layer Base Class: A foundational class for creating custom layers.
- Fully Connected Layer (FCLayer): Implements a fully connected neural network layer.
- Activation Functions: Currently supports the tanh activation function.
- Loss Functions: Implements the mean squared error (MSE) loss function.
- Activation Layer: A layer that applies an activation function to its inputs.
- Network Class: Represents a neural network. It allows adding layers, training the network, and making predictions.

## Usage
### Creating a Network:
```python
net = Network()
```
### Adding Layers:
```python 
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

## Dependencies
- numpy

## Installation
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
```bash
git clone https://github.com/Bakhitovd/IntroDL
```
```python
from ann import Network, FCLayer, ActivationLayer
```
