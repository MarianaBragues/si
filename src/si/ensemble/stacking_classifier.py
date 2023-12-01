import numpy as np
from typing import List

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class StackingClassifier:
    """
    The Stacking Classifier model harnesses an ensemble of models to generate predictions. 
    These predictions are subsequently employed to train another model – the final model. 
    The final model can then be used to predict the output variable (Y).

    Parameters
    ----------
    models:
        initial set of models

    final_model:
        the model to make the final predictions
    """
    def __init__(self, models: List, final_model):
        """
        Parameters
        ----------
        models:
            initial set of models

        final_model:
            the model to make the final predictions
        """
        self.models = models
        self.final_model = final_model
    

    def fit(self, dataset: Dataset):
        """
        Train the ensemble models

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: Stacking Classifier
            The fitted model
        """
        for md in self.models:
            md.fit(dataset)
         
        predictions = [] 
        for md in self.models:
            predictions.append(md.predict(dataset))
        
        predictions = np.array(predictions).T
        
        self.final_model.fit(Dataset(dataset.X, predictions))

        return self
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the labels using the ensemble models

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """

        initial_predictions = []
        for md in self.models:
            initial_predictions.append(md.predict(dataset))
        initial_predictions = np.array(initial_predictions).T
        final_predictions = self.final_model.predict(Dataset(dataset.X, initial_predictions))
        
        return final_predictions


    def score(self, dataset: Dataset) -> float:
        """
        Computes the accuracy between predicted and real labels

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        return accuracy(dataset.y, self.predict(dataset))
