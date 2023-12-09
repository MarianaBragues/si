#importação dos packages necessários:
from typing import Literal
import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier:
    """
    Ensemble machine learning technique that combines multiple decision trees to improve prediction accuracy 
    and reduce overfitting
    """
    #definir o construtor:
    def __init__(self, n_estimators:int = 100, 
                 max_features:int = None,
                 min_sample_split:int = 2,
                 max_depth:int = 15,
                 mode: Literal['gini','entropy'] = 'gini',
                 seed:int = None):
        """
        Random Forest is an ensemble machine learning technique that combines multiple decision trees to improve prediction 
        accuracy and reduce overfitting.
    
        parameters:
        -----------
        n_estimators: 
            number of decision trees to use
        
        max_features:
            maximum number of features to use per tree
        
        min_sample_split:
            minimum samples allowed in a split

        max_depth:
            maximum depth of the trees

        mode:
            impurity calculation mode (gini or entropy)

        seed:
            random seed to use to assure reproducibility

        estimated parameters:
        ---------------------
        trees: list
            the trees of the random forest and respective features used for training (initialized as an empty list)
        """
        #attributes
        self.n_estimators = n_estimators #atribui o número de estimadores (trees) que serão usados na forest
        self.max_features = max_features #define o número máximo de features a serem consideradas em cada tree
        self.min_sample_split = min_sample_split #especifica o número mínimo de amostras permitidas numa divisão
        self.max_depth = max_depth #define a profundidade máxima das trees da forest
        self.mode = mode #determina o modo de cálculo da impureza (pode ser 'gini' ou 'entropy')
        self.seed = seed #define a seed aleatória para garantir a reprodutibilidade

        #parameters
        self.trees = [] #inicia uma lista vazia que conterá as trees da forest
        self.training = {} #inicia um dicionário vazio para armazenar dados de treino

    def set_random_seed(self):
        """
        Set a random seed for NumPy

        Returns
        -------
        A random value
        """
        if self.seed is not None: #verifica se um valor de seed foi especificado anteriormente. Se a seed não for None 
            #(ou seja, se uma seed foi fornecida):
            np.random.seed(self.seed) #define a seed para a geração de números aleatórios na biblioteca NumPy usando o 
            #valor especificado em self.seed


    def fit(self, dataset:Dataset)->'RandomForestClassifier':
        """
        Train the decision trees of the random forest

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RandomForest
            The fitted model
        """
        self.set_random_seed() #chama a função set_random_seed() para configurar a seed de gerar números aleatórios, 
        #garantindo a reprodutibilidade dos resultados
            
        n_samples, n_features = dataset.shape() #obtém o número de amostras (n_samples - número de linhas) e o número de 
        #features (n_features - número de colunas) do dataset
        if self.max_features is None: #verifica se o número máximo de features por tree não foi especificado
            self.max_features = int(np.sqrt(n_features)) #se self.max_features for None, será definido como a raiz 
            #quadrada do número de features no dataset

        #criar a bootstrap 
        for x in range(self.n_estimators): #itera sobre o número de estimadores (self.n_estimators) especificados para a 
            #forest aleatória
            bootstrap_samples = np.random.choice(n_samples, n_samples, replace = True) #gera um conjunto de amostras de 
            #bootstrap selecionando aleatoriamente amostras do dataset, permitindo substituição - variável bootstrap_samples
            bootstrap_features = np.random.choice(n_features, self.max_features, replace=False) #gera um conjunto de features de 
            #bootstrap selecionando aleatoriamente features do dataset, sem substituição - variável bootstrap_features
            
            #cria um novo dataset (random_dataset) contendo apenas as amostras e features selecionadas aleatoriamente:
            random_dataset = Dataset(dataset.X[bootstrap_samples][:,bootstrap_features], dataset.y[bootstrap_samples])

            #variável tree - inicia um modelo de árvore de decisão com os parâmetros fornecidos:
            tree = DecisionTreeClassifier(min_sample_split=self.min_sample_split, max_depth=self.max_depth, mode = self.mode)

            tree.fit(random_dataset) #treina a árvore de decisão com o novo dataset aleatório

            self.trees.append((bootstrap_features, tree)) #adiciona a tree treinada à lista de trees (self.trees), 
            #juntamente com os índices das features usadas para o treino 

        return self
    

    def predict(self, dataset:Dataset)-> np.ndarray:
        """
        Predicts the class labels for a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset for which to make predictions.

        Returns
        -------
        np.ndarray
            An array of predicted class labels.
        """
        n_samples = dataset.shape()[0] #cria a variável n_samples que contém o número de amostras no dataset fornecido
        predictions = np.zeros((self.n_estimators, n_samples), dtype=object) #cria a variável predictions que será uma 
        #matriz de zeros para armazenar as previsões das árvores de decisão. O tamanho da matriz será 
        #(n_estimators, n_samples), onde n_estimators é o número de árvores na forest aleatória e n_samples é o número de 
        #amostras no dataset
        
        for tree_idx, (feature_idx, tree) in enumerate(self.trees): #itera sobre as árvores (tree) na lista de árvores 
            #(self.trees) juntamente com os índices das features (feature_idx) usadas no treino
            data_samples = Dataset(dataset.X[:, feature_idx], dataset.y) #cria um novo dataset contendo apenas as features
            #usadas por cada tree para fazer previsões, permitindo fazer previsões usando apenas as features relevantes para cada tree
            tree_preds = tree.predict(data_samples)  #chama a função predict de cada tree para fazer previsões com base no
            #conjunto de dados data_samples específico para aquela tree
            predictions[tree_idx, :] = tree_preds  #armazena as previsões feitas pela tree atual na matriz de previsões 
            #(predictions) na linha correspondente à tree (tree_idx)

        return predictions #devolve a matriz de previsões


    def score(self, dataset: Dataset) -> float:
        """
        Computes the accuracy between predicted and real labels

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the RandomForest on

        Returns
        -------
        random_forest: float
            The Mean Square Error of the model
        """
        #avaliar o modelo:
        predictions = self.predict(dataset) #usa a função predict para fazer previsões com base no dataset fornecido
        #Estas previsões são armazenadas na variável predictions
        return accuracy(dataset.y, predictions) #devolve o cálculo da acurácia comparando as previsões geradas pelo modelo
        #(predictions) com as labels reais do dataset (dataset.y) e devolve a precisão do modelo



#Testes:
if __name__ == '__main__':
    #criar dados aleatórios:
    np.random.seed(42)
    X = np.random.rand(100, 3)
    theta_true = np.array([3, 1.5, -2])
    noise = 0.1 * np.random.randn(100)
    y = X.dot(theta_true) + noise

    #criar o dataset:
    dataset = Dataset(X, y, features=['Feature1', 'Feature2', 'Feature3'], label='Target')

    #Testar o modelo:
    random_forest = RandomForestClassifier(n_estimators=100, max_features=None, min_sample_split=2, max_depth=15, mode='gini', seed=None)
    random_forest.fit(dataset)

    print("Trees:", len(random_forest.trees))

    predictions = random_forest.predict(dataset)
    print("Predictions for the first 5 samples:", predictions[:, :5])

    accuracy_score = random_forest.score(dataset)
    print("Accuracy:", accuracy_score)

    
