#importação dos packages necessários:
import numpy as np
from si.data.dataset import Dataset


class VarianceThreshold:
    """
    Variance Threshold feature selection.
    Features with a training-set variance lower than this threshold will be removed from the dataset.

    Parameters
    ----------
    threshold: float
        The threshold value to use for feature selection. Features with a
        training-set variance lower than this threshold will be removed.

    Attributes
    ----------
    variance: array-like, shape (n_features,)
        The variance of each feature.
    """

    def __init__(self, threshold: float = 0.0):
        #definição do construtor:
        """
        Variance Threshold feature selection.
        Features with a training-set variance lower than this threshold will be removed from the dataset.

        Parameters
        ----------
        threshold: float
            The threshold value to use for feature selection. Features with a
            training-set variance lower than this threshold will be removed.
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        # parameters
        self.threshold = threshold

        # attributes
        self.variance = None #inicia a variância como None

    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Fit the VarianceThreshold model according to the given training data.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit.

        Returns
        -------
        self : object
        """
        #calculo da variância das features do dataset de treino, através da função var do NumPy, onde axis=0 indica que a 
        #variância deve ser calculada ao longo do eixo das colunas. O resultado é armazenado no atributo variance:
        self.variance = np.var(dataset.X, axis=0)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It removes all features whose variance does not meet the threshold.
        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        dataset: Dataset
        """
        X = dataset.X #atribui o conjunto de dados de features (X) do dataset à variável X

        features_mask = self.variance > self.threshold #cria uma mask onde os valores correspondem a True se a variância 
        #da feature for maior que o threshold
        X = X[:, features_mask] #usa a mask para selecionar apenas as features cuja variância atende ao threshold, 
        #atualizando o dataset X apenas com essas features
        features = np.array(dataset.features)[features_mask] #seleciona as features correspondentes à mask, criando um 
        #novo conjunto de features
        #devolve um novo Dataset com as features selecionadas, mantendo as labels (y), a lista de features e a label do 
        #dataset original, devolve um dataset transformado:
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label)


    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit to data, then transform it.
        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        dataset: Dataset
        """
        self.fit(dataset) #chama a função fit, ajustando o modelo aos dados de entrada dataset
        return self.transform(dataset) #devolve o resultado da transformação aplicada ao dataset fornecido. 
        #Chama a função transform para aplicar a transformação ao dataset após ter sido ajustado pelo método fit


if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    selector = VarianceThreshold()
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)