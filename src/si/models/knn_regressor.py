from typing import Callable, Union
import numpy as np
from si.metrics.rmse import rmse
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor:
    """
    KNN Classifier
    The k-Nearst Neighbors classifier is a machine learning model that estimates the average value of the k most similar 
    examples instead of the most common class.

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use

    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initialize the KNN Regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self
    
    def _get_closest_label(self, x: np.ndarray):
        """
        Calculates the mean of the class with the highest frequency.

        Parameters 
            x: Array of samples.

        Returns: 
            Indexes of the classes with the highest frequency
        """
        # Calculates the distance between the samples and the dataset
        distances = self.distance(x, self.dataset.X)
        # Sort the distances and get indexes
        knn = np.argsort(distances)[:self.k]  # get the first k indexes of the sorted distances array
        knn_labels = self.dataset.y[knn]
        # Computes the mean of the matching classes
        match_class_mean = np.mean(knn_labels)
        # Sorts the classes with the highest mean
        return match_class_mean
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of (test dataset)

        Returns
        -------
        predictions: np.ndarray
            An array of predicted values for the testing dataset (Y_pred)
        """
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        Returns the accuracy of the model.
        :return: Accuracy of the model.
        """
        predictions = self.predict(dataset)

        return rmse(dataset.y, predictions)