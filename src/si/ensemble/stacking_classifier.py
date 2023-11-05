import numpy as np
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
    def __init__(self, model, final_model):
        """
        Parameters
        ----------
        models:
            initial set of models

        final_model:
            the model to make the final predictions
        """
        self.model = model
        self.final_model = final_model
    

    def fit(self, dataset: Dataset) -> 'StackingClassifier':
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
            md.fit(dataset.X, dataset.y)
         
        predictions = []
        for md in self.model:
            predictions.append(md.predict(dataset.X))
        predictions = np.vstack(predictions).T

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
        model_predictions = []
        for md in self.model:
            model_predictions.append(md.predict(dataset.X))
        model_predictions = np.vstack(model_predictions).T

        final_predictions = self.final_model.predict(model_predictions)

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

        predictions = self.predict(dataset.X)
        accuracy_score = np.mean(predictions == dataset.y)
        return accuracy_score
