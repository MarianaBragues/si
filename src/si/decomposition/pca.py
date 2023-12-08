#importação da biblioteca NumPy e da classe Dataset do módulo si.data.dataset:
import numpy as np
from si.data.dataset import Dataset


class PCA:
    """
    PCA is a linear algebra technique used to reduce the dimensions of the dataset. 
    The PCA to be implemented must use the Singular Value Decomposition (SVD) linear algebra technique.
    """
    #define o construtor da classe:
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
        #calcula a média, os componentes principais e a variância explicada do conjunto de dados
        #centra os dados, subtraindo a média:
        self.mean = np.mean(dataset.X, axis = 0) #calcula a média ao longo do longo das amostras dos dados dataset.X e 
        #armazena o resultado em mean
        dataset = dataset.X - self.mean #cria a variável dataset onde centraliza os dados subtraindo a média das amostras 
        #em dataset.X

        #calcula a SVD dos dados para obter os componentes principais:
        self.U,self.S,self.V = np.linalg.svd(dataset, full_matrices=False) #realiza a SVD dos dados centralizados para 
        #obter as matrizes U, S e V
        
        #componentes principais
        self.components = self.V[:self.n_components] #define a variável components, onde vai buscar os n_components que 
        #são os primeiros componentes da matriz de V

        #calcula a variância explicada pelos componentes principais:
        n_samples = dataset.shape[0] #cria a variável n_samples onde é calculado o número de amostras dos dados
        EV = (self.S ** 2)/(n_samples - 1) #cria a variável EV (variância explicada pelos componentes principais), 
        #dividindo o quadrado dos valores singulares pela quantidade de amostras menos 1
        self.explained_variance = EV[:self.n_components] #define os primeiros n_components dos valores de variância 
        #explicada como "explained_variance"

        return self #devolve o dataset 


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
        #centraliza os dados do dataset.X subtraindo a média que foi calculada durante o fit, garantindo assim que os dados 
        #de entrada estejam centrados antes da transformação, e guarda os valores na variável dataset
        dataset = dataset.X - self.mean
        
        #obtém a matriz transposta dos componentes principais: components contém os componentes principais calculados 
        #durante o fit.
        v_matrix = self.components.T

        #realiza a transformação dos dados: faz a multiplicação escalar de dois arrays utilizando a função "dot" da 
        #biblioteca "NumPy", ou seja, entre os dados centralizados (dataset) e a matriz transposta dos componentes 
        #principais (v_matrix), projetando os dados originais nos novos componentes principais, resultando nos dados 
        #reduzidos - esta informação fica guardada na variável reduced_data
        reduced_data = np.dot(dataset, v_matrix)

        return reduced_data #devolve os dados transformados (os dados originais reduzidos para a dimensionalidade 
        #especificada pelos componentes principais)


    def fit_transform(self, dataset: Dataset) -> Dataset:
        #combinação dos métodos fit e transform anteriormente definidos da classe PCA:
        """
        Runs fit and the transform

        Return: 
            Dataset object
        """
        self.fit(dataset) #chama o método fit, que calcula a média, os componentes principais e a variância explicada com 
        #base nos dados fornecidos (dataset)
        return self.transform(dataset) #chama o método transform, que aplica a transformação dos dados originais para os 
        #novos componentes principais calculados durante o ajuste. Devolve os dados transformados do objeto 'Dataset'


#Testes:
if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) #conjunto de dados como exemplo
    dataset = Dataset(X, None, None, None)

    pca = PCA(n_components=2) #PCA com dois componentes
    
    #Teste ao método fit:
    pca.fit(dataset)
    print("Mean:", pca.mean)
    print("Components:", pca.components)
    print("Explained Variance:", pca.explained_variance)

    #Teste ao método transform
    transformed_data = pca.transform(dataset)
    print("Transformed Data:")
    print(transformed_data)

    #Testa ao método fit_transform
    transformed_dataset = pca.fit_transform(dataset)
    print("Transformed Dataset:")
    print(transformed_dataset)