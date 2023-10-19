import numpy as np

from si.data.dataset import Dataset


class PCA:
    def __init__(self, n_components: int = 2):
        """
        Initializes the PCA.
        :param n_components: Number of components to keep.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _get_centered_data(self, dataset: Dataset) -> np.ndarray:
        """
        Centers the dataset.
        :param dataset: Dataset object.
        :return: A matrix with the centered data.
        """

        self.mean = np.mean(dataset.X, axis=0)  # axis=0 means that we want to calculate the mean for each column
        self.centered_data = dataset.X - self.mean
        return self.centered_data

    def _get_components(self, dataset: Dataset) -> np.ndarray:
        """
        Calculates the components of the dataset.
        :param dataset:
        :return: A matrix with the components.
        """

        # Get centered data
        centered_data = self._get_centered_data(dataset)

        # Get single value decomposition
        self.u_matrix, self.s_matrix, self.v_matrix_t = np.linalg.svd(centered_data, full_matrices=False)

        # Get principal components
        self.components = self.v_matrix_t[:, :self.n_components]  # get the first n_components columns

        return self.components

    def _get_explained_variance(self, dataset: Dataset) -> np.ndarray:
        """
        Calculates the explained variance.
        :param dataset: Dataset object.
        :return: A vector with the explained variance.
        """
        # Get explained variance
        ev_formula = self.s_matrix ** 2 / (len(dataset.X) - 1)
        explained_variance = ev_formula[:self.n_components]

        return explained_variance

    def fit(self, dataset: Dataset):
        """
        Calculates the mean, the components and the explained variance.
        :return: Dataset.
        """

        self.components = self._get_components(dataset)
        self.explained_variance = self._get_explained_variance(dataset)

        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset.
        :return: Dataset object.
        """
        if self.components is None:
            raise Exception("You must fit the PCA before transform the dataset.")

        # Get transposed V matrix
        v_matrix = self.v_matrix_t.T

        # Get transformed data
        transformed_data = np.dot(self.centered_data, v_matrix)

        return Dataset(transformed_data, dataset.y, dataset.features, dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Calculates the mean, the components and the explained variance and transforms the dataset.
        :return: Dataset object.
        """
        self.fit(dataset)
        return self.transform(dataset)