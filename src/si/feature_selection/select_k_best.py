from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest:
    """
    Select features according to the k highest scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    k: int, default=10
        Number of top features to select.

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """
    def __init__(self, score_func: Callable = f_classification, k: int = 10):
        #define o construtor com os parâmetros iniciais
        """
        Select features according to the k highest scores.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        k: int, default=10
            Number of top features to select.
        """
        self.k = k #número de features a serem selecionadas, by default é 10
        self.score_func = score_func #calcula algum tipo de pontuação ou valor para as features, by default é f_classification
        self.F = None
        self.p = None
        #as variáveis F e p são inicializadas como None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        It fits SelectKBest to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset) #chama a função de pontuação (score_func), passando o dataset fornecido 
        #como argumento. Esta função devolve os valores de pontuação de F e p para as features no dataset. 
        return self #devolve o dataset com as novas alterações

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the k highest scoring features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the k highest scoring features.
        """
        idxs = np.argsort(self.F)[-self.k:] #usa np.argsort para obter os índices das K features de maior pontuação nos 
        #valores de F armazenados em F
        features = np.array(dataset.features)[idxs] #usa os índices selecionados para obter os nomes das features 
        #correspondentes no dataset de entrada
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label) #devolve um novo 
        #dataset contendo apenas as features selecionadas com base nos índices obtidos. Seleciona as colunas correspondentes
        #nos dados de entrada X, atualiza as labels das features para as selecionadas, mantém as labels y e label no novo 
        #conjunto de dados.

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectKBest and transforms the dataset by selecting the k highest scoring features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the k highest scoring features.
        """
        self.fit(dataset) #chama o método fit desta classe que calcula os valores de pontuação de F para as features no 
        #dataset fornecido
        return self.transform(dataset) #chama o método transform desta classe e seleciona as features com base nos valores
        #de F calculados anteriormente. Devolve um novo dataset contendo apenas as K features de maior pontuação

#Testes:
if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    selector = SelectKBest(k=2)
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)