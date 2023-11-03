import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares:
# implement class
    """
    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    scale: bool
        Wheter to scale the data or not 

    Attributes
    ----------
    theta: np.array
        the coefficients of the model for every feature
    theta_zero: float
        the zero coefficient (y intercept) 
    mean: np.ndarray
        mean of the dataset (for every feature)
    std: np.ndarray
        standard deviation of the dataset (for every feature) 
    """
    def __init__(self, l2_penalty: float, scale: bool = True):
        """
        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        scale: bool
            Whether to scale the dataset or not
        """
        #parameters
        self.l2_penalty = l2_penalty
        self.scale = scale

        #attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fit the model to the dataset by estimating the theta and theta_zero,
        coefficients, mean and std

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegressionLeastSquares
            The fitted model
        """
        if self.scale:
            #compute mean and std
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            #scale the dataset
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = dataset.shape()

        #add a column of ones in the first column position
        modified_X = np.c_[np.ones(m), X]

        #compute the penalty matrix (term l2_penalty * identity matrix)
        identity_matrix = np.eye(n + 1)
        penalty_matrix = self.l2_penalty * identity_matrix

        #change the first position of the penalty matrix to 0
        penalty_matrix[0,0] = 0

        #compute the model parameters 
        matrix1 = modified_X.T
        matrix2 = matrix1.dot(dataset.y)

        #calculate the inverse matrix
        inverse_matrix = np.linalg.inv(matrix1.dot(modified_X) + penalty_matrix)

        #calculate the thetas
        thetas_final = inverse_matrix.dot(matrix2)

        self.theta = thetas_final[1:]
        self.theta_zero = thetas_final[0]
        
        return self
    
    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X

        #add a column of ones in the first column position
        m, n = dataset.shape()
        modified_X = np.c_[np.ones(m), X]

        #concatenate theta_zero and theta
        conc_thetas = np.r_[self.theta_zero, self.theta]
        #matrix multiplication
        y_pred = modified_X.dot(conc_thetas)
        return y_pred
    
    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        predicted_y = self.predict(dataset)
        return mse(dataset.y, predicted_y)
