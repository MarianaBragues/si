#importação dos packages necessários:
import numpy as np
from typing import List
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class StackingClassifier:
    """
    The Stacking Classifier model harnesses an ensemble of models to generate predictions. 
    These predictions are subsequently employed to train another model – the final model. 
    The final model can then be used to predict the output variable (Y).

    Parameters
    ----------
    models:
        initial set of models

    final_model:
        the model to make the final predictions
    """
    def __init__(self, models: List, final_model):
        #define o construtor
        """
        Parameters
        ----------
        models:
            initial set of models

        final_model:
            the model to make the final predictions
        """
        self.models = models #lista o conjunto inicial de modelos
        self.final_model = final_model #modelo final que utiliza as previsões dos modelos iniciais para fazer a previsão final
    

    def fit(self, dataset: Dataset):
        """
        Train the ensemble models

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: Stacking Classifier
            The fitted model
        """
        for md in self.models: #inicia um loop sobre os modelos (md) presentes na lista self.models (modelos iniciais)
            md.fit(dataset) #para cada modelo (md), chama a função fit passando o dataset de treino, treina cada modelo 
            #inicial com os dados fornecidos
         
        predictions = [] #cria a variável predictions, uma lista vazia, para armazenar as previsões dos modelos iniciais
        for md in self.models: #inicia um loop sobre os modelos (md) presentes na lista self.models
            predictions.append(md.predict(dataset)) #para cada modelo (md), chama a função predict passando o dataset de 
            #treino. As previsões de cada modelo são adicionadas à lista predictions
        
        predictions = np.array(predictions).T #converte a lista de previsões num array NumPy e realiza a transposta (T) 
        #para reorganizar as previsões de modo que cada linha represente uma amostra e cada coluna seja a previsão de um 
        #modelo
        
        self.final_model.fit(Dataset(dataset.X, predictions)) #utiliza o array de previsões para treinar o modelo final 
        #(self.final_model). Cria um novo dataset onde as previsões dos modelos iniciais são os recursos para treinar o 
        #modelo final

        return self
    

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the labels using the ensemble models

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        initial_predictions = [] #cria a variável initial_predictions, uma lista vazia, para armazenar as previsões dos modelos iniciais
        for md in self.models: #cria um loop que itera sobre os modelos presentes na lista self.models
            initial_predictions.append(md.predict(dataset)) #para cada modelo (md), a função predict é chamada, passando o
            #dataset de teste. As previsões de cada modelo são adicionadas à lista initial_predictions
        
        initial_predictions = np.array(initial_predictions).T #a lista de previsões é convertida num array NumPy e é 
        #feita a transposta (T), organizando as previsões de modo que cada linha represente uma amostra e cada coluna seja
        #a previsão de um modelo
        
        final_predictions = self.final_model.predict(Dataset(dataset.X, initial_predictions)) #o modelo final (self.final_model) 
        #é usado para fazer previsões com base nas previsões combinadas dos modelos iniciais. É criado um novo dataset onde as
        #previsões combinadas dos modelos iniciais são tratadas como recursos para prever com o modelo final.
        
        return final_predictions #devolve as previsões finais do modelo


    def score(self, dataset: Dataset) -> float:
        """
        Computes the accuracy between predicted and real labels

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        #chama a função predict para fazer previsões no dataset, em seguida, essas previsões são comparadas com as labels 
        #reais (dataset.y) usando a função accuracy, devolvendo a acurácia do modelo
        return accuracy(dataset.y, self.predict(dataset))


#Testes
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression

if __name__ == '__main__':
    #criar um conjunto de dados de exemplo
    np.random.seed(42)
    X_train = np.random.rand(100, 3)  #matriz de 100 amostras e 3 features
    y_train = np.random.randint(0, 2, size=100)  #valores de classe (0 ou 1)
    X_test = np.random.rand(50, 3)  #matriz de teste com 50 amostras
    y_test = np.random.randint(0, 2, size=50)  #valores de classe para teste (0 ou 1)

    #criar um Dataset para treino e teste
    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)

    #criar os modelos iniciais
    decision_tree = DecisionTreeClassifier()
    knn = KNNClassifier()
    logistic_regression = LogisticRegression()

    #criar o modelo final
    final_model = DecisionTreeClassifier()

    #criar o StackingClassifier com os modelos iniciais e o modelo final
    models = [decision_tree, knn, logistic_regression]
    stacking_classifier = StackingClassifier(models, final_model)

    #treinar o StackingClassifier
    stacking_classifier.fit(train_dataset)

    #faz previsões no conjunto de teste
    predictions = stacking_classifier.predict(test_dataset)

    #avaliar o modelo
    acc = accuracy(test_dataset.y, predictions)
    print(f'StackingClassifier accuracy: {acc:.4f}')