import numpy as np
from typing import List
from si.model_selection.split import train_test_split
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
        self.final_model_trained = False
    

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
        for md in self.model:
            md.fit(dataset)
         
        predictions = np.array([md.predict(dataset) for md in self.models]).T

        self.final_model.fit(Dataset(X=predictions, y=dataset.y))
        self.final_model_trained = True

        self.final_model.fit(predictions, dataset.y)
        
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
        if not self.final_model_trained:
            raise ValueError("Final model has not been trained.")

        initial_predictions = np.array([model.predict(dataset) for model in self.models]).T
        final_predictions = self.final_model.predict(Dataset(X=initial_predictions))

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

        predictions = self.predict(dataset)
        accuracy_score = accuracy(dataset.y, predictions)
        return accuracy_score
