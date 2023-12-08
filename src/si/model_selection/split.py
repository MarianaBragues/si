#importação dos packages necessários:
from typing import Tuple
import numpy as np
from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    #função recebe um objeto Dataset para dividir, um parâmetro test_size que define a proporção do conjunto de teste 
    #(default 0.2 - 20%) e um parâmetro random_state para controlar a aleatoriedade da divisão. 
    #devolve um par de datasets, um para treino e outro para teste.
    np.random.seed(random_state) #gere números aleatórios do NumPy para garantir a reprodutibilidade dos resultados
    n_samples = dataset.shape()[0] #cria a variável n_samples onde é calculado o número de amostras no conjunto de dados 
    n_test = int(n_samples * test_size) #cria a variável n_test que determina o tamanho do conjunto de teste com base na 
    #proporção especificada
    permutations = np.random.permutation(n_samples) #cria a variável permutations que gera uma permutação aleatória dos 
    #índices das amostras nos dados
    test_idxs = permutations[:n_test] #cria a variável test_idxs que usa os índices permutados para selecionar amostras 
    #para o conjunto de teste com base no tamanho calculado para o conjunto de teste
    train_idxs = permutations[n_test:] #cria a variável train_idxs que usa os índices permutados para selecionar amostras 
    #para o conjunto de treino com base no tamanho calculado para o conjunto de teste
    #criação dos conjuntos de dados separados (train e test) usando os índices selecionados para as amostras de treino e teste
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test #devolve os datasets de treino e teste (tuple)

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into stratified training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    #função recebe um objeto Dataset para dividir, um parâmetro test_size para determinar a proporção do conjunto de teste
    #(default 0.2 - 20%) e um parâmetro random_state para controlar a aleatoriedade da divisão. Devolve um par de datasets,
    #um para treino e outro para teste, mantendo a estratificação das classes.
    unique_labels, label_counts = np.unique(dataset.y, return_counts=True) #obtém as classes únicas presentes nos rótulos 
    #(y) do dataset e conta quantas vezes cada classe aparece
    #inicializa listas vazias para armazenar os índices das amostras que serão selecionadas para os conjuntos de treino e teste:
    train_indices = [] #lista vazia para os dados de treino
    test_indices = [] #lista vazia para os dados de teste

    np.random.seed(random_state) #gera uma permutação aleatória do NumPy para garantir a reprodutibilidade dos resultados

    for label, count in zip(unique_labels, label_counts): #iteração sobre as classes únicas e as suas contagens para 
        #realizar a estratificação das amostras
        num_test_samples = int(count * test_size) #cria a variável num_test_samples que calcula o número de amostras de 
        #teste para a classe atual com base na proporção especificada
        class_indices = np.where(dataset.y == label)[0]
        np.random.seed(random_state) 
        np.random.shuffle(class_indices) #baralha os índices das amostras da classe atual 
        test_indices.extend(class_indices[:num_test_samples]) #seleciona as amostras necessárias para o conjunto de teste,
        #adicionando esses índices à lista de índices de teste
        train_indices.extend(class_indices[num_test_samples:]) #adiciona os índices restantes (após a seleção para o 
        #conjunto de teste) à lista de índices de treino
    #criação dos datasets separados (train_dataset e test_dataset) usando os índices selecionados para as amostras de 
    #treino e teste:
    train_dataset = Dataset(dataset.X[train_indices], dataset.y[train_indices], features=dataset.features, label=dataset.label)
    test_dataset = Dataset(dataset.X[test_indices], dataset.y[test_indices], features=dataset.features, label=dataset.label)

    return train_dataset, test_dataset #devolve os datasets de treino e teste (tuple)


#Testes:
if __name__ == '__main__':
    #conjunto de dados de exemplo:
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([0, 1, 0, 1])
    features = np.array(['a', 'b', 'c'])
    label = 'target'
    dataset = Dataset(X, y, features, label)

    #Teste da função train_test_split
    train, test = train_test_split(dataset, test_size=0.25, random_state=42)
    print("Train Set:")
    print(train.X)
    print(train.y)
    print("Test Set:")
    print(test.X)
    print(test.y)

    #Teste da função stratified_train_test_split
    strat_train, strat_test = stratified_train_test_split(dataset, test_size=0.25, random_state=42)
    print("Stratified Train Set:")
    print(strat_train.X)
    print(strat_train.y)
    print("Stratified Test Set:")
    print(strat_test.X)
    print(strat_test.y)
             