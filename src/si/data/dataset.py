from typing import Tuple, Sequence, Union

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None) -> None: 
        #construtor da classe que recebe a matriz X e o vetor y, uma lista de features e uma label
        """
        Dataset represents a tabular dataset for single output classification.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        #Faz várias verificações para garantir a integridade dos dados 
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        #Atribui os valores aos atributos da classe:
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape #este método retorna um tuple no formato (n_samples, n_features), onde n_samples é o número 
    #de amostras (linhas) e n_features é o número de características (colunas) no conjunto de dados.

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        return self.y is not None #verifica se o conjunto de dados tem uma label associado, verificando se o atributo 
    #self.y (que representa as labels) não é None. Retorna True se self.y não for None, indicando que há labels presentes 
    #no conjunto de dados. Retorna False se self.y for None, indicando a ausência de labels no conjunto de dados.

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.has_label(): #verifica se o conjunto de dados tem uma label associada usando a função has_label()
            return np.unique(self.y) #se tiver label (self.y não é None), utiliza a função np.unique(self.y) para retornar
            #um array NumPy contendo as classes únicas presentes nas label
        else:
            raise ValueError("Dataset does not have a label") #se não houver label (self.y is None), surge uma exceção 
        #ValueError indicando que o conjunto de dados não possui labels

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0) #calcula a média ao longo do eixo das features (axis=0), usando a função 
    #np.nanmean (média é calculada considerando apenas os valores numéricos presentes em cada coluna)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0) #utiliza a função np.nanvar (variância é calculada considerando apenas os valores
        #numéricos presentes em cada coluna) para calcular a variância ao longo do eixo das features (axis=0) 

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0) #utiliza a função np.nanmedian (a mediana é calculada considerando apenas os 
    #valores numéricos presentes em cada coluna) para calcular a mediana ao longo do eixo das features (axis=0) 

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0) #utiliza a função np.nanmin (são considerados apenas os valores númericos 
    #presentes em cada coluna) para devolver o valor mínimo ao longo do eixo das features (axis=0) 

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)  #utiliza a função np.nanmax (são considerados apenas os valores númericos 
    #presentes em cada coluna) para devolver o valor máximo ao longo do eixo das features (axis=0) 

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        #cria um dicionário 'data' com o resumo estatístico onde a chave é "mean", "median", "min", "max", "var" e o valor
        #associado é o resultado desse cálculo
        data = { 
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features) #cria um dataframe do Pandas apartir do
    #dicionário data. As linhas do DataFrame são "mean", "median", "min", "max", "var" e as colunas são as features, com 
    #os nomes fornecidos em self.features.

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label: #se uma label é especificada:
            X = df.drop(label, axis=1).to_numpy() #cria X removendo a coluna da label do dataframe
            y = df[label].to_numpy() #cria y apenas com os valores da coluna da label
        else: #se nenhuma label for especificada:
            X = df.to_numpy() #cria X usando todos os dados do dataframe
            y = None #y é none

        features = df.columns.tolist() #as features são obtidas a partir dos nomes das colunas do dataframe
        return cls(X, y, features=features, label=label) #devolve um dataset com os dados X, y, features e label

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None: #verifica se existem labels associadas ao conjunto de dados y, se não houver:
            return pd.DataFrame(self.X, columns=self.features) #devolve um dataframe usando apenas os dados X e as features
        else: #senão
            df = pd.DataFrame(self.X, columns=self.features) #cria uma variável df contendo um dataframe com os dados X e as features
            df[self.label] = self.y #adiciona as labels y ao dataframe
            return df #devolve o df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features) #cria a matriz X com dados aleatórios
        y = np.random.randint(0, n_classes, n_samples) #cria a matriz y com dados aleatórios
        return cls(X, y, features=features, label=label) #devolve o dataframe com os dados gerados
    
### 2.1) Add a method to the Dataset class that removes all samples containing at least one null value NaN. 
# Note that the resulting object should not contain null values in any independent feature variable. 
# Also, note that you should update the y vector by removing entries associated with the samples to be removed. 
# You should use only NumPy functions Method name: dropna
    def dropna(self) -> 'Dataset':
        """
        Removes samples with NaN values.
        :return: Dataset
        """
        nan_values = np.isnan(self.X).any(axis=1) #cria a variável nan_values onde verifica se existem valores NaN em 
        #self.X e retorna uma matriz booleana indicando as amostras que possuem pelo menos um valor NaN nas features
        self.X = self.X[~nan_values] # mantém apenas as amostras que não possuem NaN nas suas features

        if self.has_label(): #se o conjunto de dados tiver labels (recorre à função has_label() 
            self.y = self.y[~nan_values] #remove as labels associadas às amostras removidas, mantendo apenas as labels correspondentes às amostras mantidas.

        return self #devolve o dataset
    
    
### 2.2) Add a method to the Dataset class that replaces all null values with another value or the mean or median of the 
# feature/variable. Note that the resulting object should not contain null values in any independent feature/variable. 
# You should use only NumPy functions. Method name: fillna
    def fillna(self, values: list[float]) -> 'Dataset':
        """
        Replaces all null values with another value or the mean or median of the feature/variable
        
        Parameters:
        -----------
        values: list of medians or means to use as replacements for NaN values
        
        Return:
        Modified Dataset 
        """
        #verifica se todos os valores em values são do tipo int ou float. Se algum valor não for, cria um ValueError
        if not all(isinstance(value, (int, float)) for value in values): 
            raise ValueError("value could be only float")
        num_columns = self.X.shape[1]
        #verifica se o número de valores em values é pelo menos igual ao número de colunas em self.X, se não for cria um ValueError
        if len(values) < num_columns:
            raise ValueError("values have at least as many values as columns in X")
        #verifica se os valores em values correspondem à média ou à mediana das features atuais do conjunto de dados. 
        #Se não corresponderem, cria um ValueError.
        if not np.array_equal(values, self.get_mean()) and not np.array_equal(values, self.get_median()):
            raise ValueError("values are the array of means or medians of the variables")

        for cols in range(num_columns): #para cada coluna
            col_values = self.X[:, cols] #cria a variável col_values com todas as linhas e a coluna cols de X
            nan_v = np.isnan(col_values) #cria a nan_v onde verifica quais valores são NaN na coluna atual

            if np.any(nan_v): #se houver valores NaN
                replace_value = values[cols] #substitui esses valores por valores especificados em values para essa coluna
                col_values[nan_v] = replace_value
                self.X[:, cols] = col_values #atualiza a coluna no conjunto de dados com os valores substituídos.

        return self #devolve o dataset com NaN substituídos por valores específicos
    

### 2.3) Add a method to the Dataset class that removes a sample by its index. Note that you should also update the 
# y vector by removing the entry associated with the sample to be removed. You should use only NumPy functions. 
# Method name: remove_by_index
    def remove_from_index(self, index: int) -> 'Dataset':
            """
            Removes a sample by index
            
            Parameters:
            -----------
            index: integer corresponding to the sample to remove
            
            Return:
            Modified Dataset 
            """
            if index < 0 or index >= len(self.X): #verifica se o índice fornecido está dentro dos limites válidos para o 
                #conjunto de dados. Se o índice estiver fora desses limites, cria um ValueError
                raise ValueError("Index is not valid, it is out of bounds")

            self.X = np.delete(self.X, index, axis=0) #remove a amostra correspondente ao índice fornecido da matriz de 
            #features X. A função np.delete() do NumPy remove a linha especificada pelo índice da matriz X ao longo das linhas

            if self.has_label(): #usa a função has_label() para verificar se há labels associadas ao conjunto de dados
                self.y = np.delete(self.y, index) #se houver labels remove a label associada à amostra removida da matriz das 
                #labels y

            return self #devolve o dataset com as alterações

#Testes:
if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())
