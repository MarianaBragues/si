from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.metrics.mse import mse, mse_derivative
from si.neural_networks.layers import Layer


class NeuralNetwork:
    def __init__(self, epochs: int, batch_size: int, optimizer: any, learning_rate: float, verbose: bool, loss: any, metric: any, **kwargs: any) -> None:
        """
        Parameters:
        -----------
        epochs: int
            t he number of epochs to train the neural network;
        batch_size: int
            t he batch size to use for training the neural network;
        optimizer: any 
            t he optimizer to use for training the neural network;
        learning_rate: float
            t he learning rate to use for training the neural network;
        verbose: bool
            w hether to print the loss and the metric at each epoch
        loss: any
            t he loss function to use for training the neural network;
        metric: any
            t he metric to use for training the neural network;
        kwargs: any
            a dditional keyword arguments passed to the optimizer.

        Estimated parameters
        --------------------
        layers
            the layers of the NN
        history
            dictionary containning the loss and metric at each epoch
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.loss = loss
        self.metric = metric
        self.kwargs = kwargs

        self.layers = []
        self.history = {'loss': [], 'metric': []}

    def add(self, layer: list) -> list:
        if len(self.layers) == 0:
            self.layers.append(layer)
        else:
            layer.set_input_shape(self.layers[-1].output_shape())
            if hasattr(layer, 'initialize'):
                layer.initialize(self.optimizer, **self.kwargs)
            self.layers.append(layer)

    def fit(self, dataset: Dataset) -> 'NeuralNetwork':
        for epoch in range(self.epochs):
            for batch_data in dataset.batch_iterator(self.batch_size):
                inputs, targets = batch_data
                # Forward propagation
                output = inputs
                for layer in self.layers:
                    output = layer.forward_propagation(output, training=True)

                # Compute loss derivative (error)
                loss_derivative = self.loss.derivative(targets, output)

                # Backpropagation
                error = loss_derivative
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error)

                # Update weights using optimizer
                for layer in self.layers:
                    if hasattr(layer, 'update'):
                        layer.update(self.optimizer)

            # Compute loss and metric for the epoch
            epoch_loss = self.loss.loss(targets, output)
            epoch_metric = self.metric(targets, output)

            # Save loss and metric in history
            self.history['loss'].append(epoch_loss)
            self.history['metric'].append(epoch_metric)

            # Print if verbose is True
            if self.verbose:
                print(f"Epoch {epoch + 1}: Loss - {epoch_loss}, Metric - {epoch_metric}")

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the classes of the dataset.
        """
        predictions = []
        for data in dataset:
            output = data
            for layer in self.layers:
                output = layer.forward_propagation(output, training=False)
            predictions.append(output)
        return predictions
        

    def score(self, dataset: Dataset, score_func: Callable = accuracy) -> float:
        """
        Returns the accuracy of the model.
        """
        inputs, targets = dataset
        predictions = self.predict(inputs)
        return self.metric(targets, predictions)
