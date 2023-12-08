#importação dos packages necessários:
from typing import Callable
import numpy as np
from si.metrics.rmse import rmse
from si.data.dataset import Dataset
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
        #definição do construtor:
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
        self.dataset = None #dataset inicializado como None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        #função fit recebe como parâmetro um objeto Dataset para ajustar o modelo
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
        self.dataset = dataset #atribui o dataset fornecido ao atributo self.dataset da classe KNNRegressor
        return self #devolve o self após ajustar os dados
    
    def _get_closest_label(self, x: np.ndarray):
        """
        Calculates the mean of the class with the highest frequency.

        Parameters 
            x: Array of samples.

        Returns: 
            Indexes of the classes with the highest frequency
        """
        distances = self.distance(x, self.dataset.X) #cria a variável distances que guarda o valor do cálculo das 
        #distâncias entre a amostra x e todas as amostras do conjunto de dados de treino (self.dataset.X), através da 
        #distância definida no construtor (self.distance) 
        
        knn = np.argsort(distances)[:self.k] #ordena as distâncias e seleciona os índices dos k neighbors mais próximos
        #usa a função np.argsort do NumPy para devolver os índices que classificariam o array de distâncias em ordem 
        #crescente e através do [:self.k] obtém os primeiros k índices do array ordenado
        
        knn_labels = self.dataset.y[knn] #cria a variável knn_lables que contém as labels das classes correspondentes 
        #aos neighbors mais próximos determinados no passo anterior
        
        match_class_mean = np.mean(knn_labels) #cria a variável match_class_mean que contém o resultado do cálculo da 
        #média das labels das classes dos neighbors mais próximos

        return match_class_mean #devolve a média determinada
    
    
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
        #utiliza a função apply_along_axis do NumPy para aplicar a função _get_closest_label definida anteriormente 
        #ao longo do eixo 1 (linhas) do array dataset.X. Assim, a função _get_closest_label será aplicada a cada linha 
        #(amostra) do dataset de teste para calcular as previsões
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X) #devolve um array NumPy contendo os 
        #valores previstos para o dataset de teste (Y_pred)

    def score(self, dataset: Dataset) -> float:
        """
        Returns the accuracy of the model.
        :return: Accuracy of the model.
        """
        predictions = self.predict(dataset) #cria a variável predictions onde chama a função predict para obter as 
        #previsões do modelo para dataset fornecido

        return rmse(dataset.y, predictions) #Calcula o RMSE entre os valores reais (dataset.y) e as previsões (predictions) 
        #através da função rmse. Devolve o valor RMSE como a pontuação do modelo


#Testes:
if __name__ == '__main__':
    #criar os dados para treino e teste
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # Dados de treino
    y_train = np.array([10, 20, 30, 40])  # Labels de treino
    X_test = np.array([[1.5, 2.5], [3.5, 4.5]])  # Dados de teste
    y_test = np.array([15, 35])  # Labels de teste

    #criar um objeto Dataset para o conjunto de dados de treino e de teste
    dataset_train = Dataset(X_train, y_train, None, None)
    dataset_test = Dataset(X_test, y_test, None, None)

    #Teste com diferentes configurações de K
    for k_value in [1, 3]:
        for distance_func in [euclidean_distance]:  
            #inicia o modelo KNNRegressor com a configuração atual
            model = KNNRegressor(k=k_value, distance=distance_func)

            #ajusta o modelo aos dados de treino
            model.fit(dataset_train)

            #faz previsões para os dados de teste
            predictions = model.predict(dataset_test)

            #avalie o desempenho do modelo com a métrica RMSE
            score = model.score(dataset_test)
            
            #imprima os resultados
            print(f"K = {k_value}, Distance function = {distance_func.__name__}")
            print(f"Predictions: {predictions}")
            print(f"RMSE Score: {score}")
