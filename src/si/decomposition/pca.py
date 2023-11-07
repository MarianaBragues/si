import numpy as np

from si.data.dataset import Dataset


class PCA:
    """
    PCA is a linear algebra technique used to reduce the dimensions of the dataset. 
    The PCA to be implemented must use the Singular Value Decomposition (SVD) linear algebra technique.
    """
    def __init__(self, n_components: int):
        """
        Initializes the PCA.

        Parameters
        ---------- 
        n_components: int
            Number of components to keep.
        
        Estimated Parameters
        --------------------
        mean:
            mean of the samples
        components:
            the principal components (the unitary matrix of eigenvectors)
        explained_variance:
            explained variance (diagonal matrix of eigenvalues)
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None


    def fit(self, dataset: Dataset) -> np.ndarray:
        """
        Estimates the mean, principal components and explained variance.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        
        Returns
            self
        """
        #centering the data
        self.mean = np.mean(dataset.X, axis = 0)
        dataset = dataset.X - self.mean

        #calculate of SVD
        self.U,self.S,self.V = np.linalg.svd(dataset, full_matrices=False) 
        
        #infer the Principal Components
        self.components = self.V[:self.n_components]

        #infer the Explained Variance
        n_samples = dataset.shape[0]
        EV = (self.S ** 2)/(n_samples - 1)
        self.explained_variance = EV[:self.n_components]

        return self
    
    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        Transforms dataset by calculating the reduction of X to the principal components.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        
        Returns
            Reduced Dataset
        """
        #centering the data
        dataset = dataset.X - self.mean
        
        #get transposed V matrix
        v_matrix = self.components.T

        #get transformed data
        reduced_data = np.dot(dataset, v_matrix)

        return reduced_data


    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Runs fit and the transform

        Return: 
            Dataset object
        """
        self.fit(dataset)
        return self.transform(dataset)