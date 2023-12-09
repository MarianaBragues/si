#importação dos packages necessários:
import numpy as np
from si.data.dataset import Dataset
from si.metrics.mse import mse

class RidgeRegressionLeastSquares:
    """
    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    scale: bool
        Wheter to scale the data or not 

    Attributes
    ----------
    theta: np.array
        the coefficients of the model for every feature
    theta_zero: float
        the zero coefficient (y intercept) 
    mean: np.ndarray
        mean of the dataset (for every feature)
    std: np.ndarray
        standard deviation of the dataset (for every feature) 
    """
    #define o construtor da classe:
    def __init__(self, l2_penalty: float, scale: bool = True):
        """
        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        scale: bool
            Whether to scale the dataset or not
        """
        #parameters
        self.l2_penalty = l2_penalty
        self.scale = scale
        #inicia os atributos como none:
        #attributes:
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fit the model to the dataset by estimating the theta and theta_zero,
        coefficients, mean and std

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegressionLeastSquares
            The fitted model
        """
        if self.scale: #se self.scale for verdadeiro, calcula a média e o desvio padrão das features
            self.mean = np.nanmean(dataset.X, axis=0) #cálculo da média de cada coluna (feature)
            self.std = np.nanstd(dataset.X, axis=0) #cãlculo do desvio padrão de cada coluna (feature) 
            
            X = (dataset.X - self.mean) / self.std #standardização do dataset: subtrai a média e divide pelo desvio padrão
        else:
            X = dataset.X #santandardização do dataset

        m, n = dataset.shape() #obtém o número de linhas (variável m) e colunas (variável n) do dataset

        modified_X = np.c_[np.ones(m), X] #adiciona uma coluna de 1 (constante) à esquerda dos dados de entrada (X)

        identity_matrix = np.eye(n + 1) #cria a matriz de identidade: usa a função eye do NumPy (n + 1) x (n + 1).
        penalty_matrix = self.l2_penalty * identity_matrix #cria a matriz de penalidade: multiplica a matriz identidade 
        #por um fator de regularização (self.l2_penalty)

        penalty_matrix[0,0] = 0 #define o elemento na posição (0, 0) da matriz de penalidade como zero
 
        matrix1 = modified_X.T #cria a variável matrix1 que guarda o cálculo da transposta da matriz de features
        matrix2 = matrix1.dot(dataset.y) #cria a variável matrix2 que realiza a multiplicação entre a transposta de 
        #modified_X e o vetor de labels (y) do dataset

        inverse_matrix = np.linalg.inv(matrix1.dot(modified_X) + penalty_matrix) #cria a variável inverse_matrix que guarda
        #o cálculo da inversa da soma entre o produto de modified_X pela sua transposta e a penalty_matrix

        thetas_final = inverse_matrix.dot(matrix2) #cria a variável thetas_final onde calcula os coeficientes finais 
        #(theta) através da multiplicação da matriz inversa pela matriz de labels 

        #atribui os coeficientes calculados (theta e theta_zero) aos atributos do modelo:
        self.theta = thetas_final[1:]
        self.theta_zero = thetas_final[0]
        
        return self #devolve o modelo ajustado com os parâmetros calculados
    
    
    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        #se self.scale for verdadeiro, os dados são padronizados subtraindo a média e dividindo pelo desvio padrão,
        #caso contrário, os dados não são modificados:
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X

        m, n = dataset.shape() #obtém o número de linhas (m) e colunas (n) dos dados do dataset
        modified_X = np.c_[np.ones(m), X] #adiciona uma coluna de 1 (constante) à esquerda dos dados de entrada (X)

        conc_thetas = np.r_[self.theta_zero, self.theta] #cria a variável conc_thetas que concatena o coeficiente 
        #theta_zero e os coeficientes theta num único vetor, através do NumPy
        
        y_pred = modified_X.dot(conc_thetas) #faz a multiplicação entre os dados modificados e os coeficientes concatenados 
        #para fazer as previsões -> origina uma matriz

        return y_pred #devolve as previsões do modelo para o dataset fornecido
    
    
    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        #avaliação do modelo:
        predicted_y = self.predict(dataset) #utiliza a função predict para gerar as previsões para dataset fornecido e guarda
        #o resultado na variável predicted_y
        return mse(dataset.y, predicted_y) #devolve o cálculo do erro quadrático médio (MSE) comparando as previsões 
        #(predicted_y) geradas pelo modelo com os valores reais (dataset.y) usando a função mse -> desempenho do modelo


#Testes:
if __name__ == '__main__':
    #criar dados de exemplo:
    np.random.seed(42)
    X = np.random.rand(100, 3)  #matriz de 100 amostras e 3 features
    theta_true = np.array([3, 1.5, -2])  #coeficientes verdadeiros
    noise = 0.1 * np.random.randn(100)  #noise
    y = X.dot(theta_true) + noise  #gera os valores y usando os coeficientes e adicionando noise

    #criar um objeto Dataset:
    dataset = Dataset(X, y, features=['Feature1', 'Feature2', 'Feature3'], label='Target')

    #testa a classe RidgeRegressionLeastSquares:
    ridge_reg = RidgeRegressionLeastSquares(l2_penalty=0.1)
    ridge_reg.fit(dataset)

    #verifica os coeficientes estimados:
    print("Estimated Coefficients:", ridge_reg.theta)
    print("Estimated Intercept:", ridge_reg.theta_zero)

    #faz previsões:
    y_pred = ridge_reg.predict(dataset)
    print("Predictions:", y_pred[:5])  #exibe as cinco primeiras previsões

    #calculando o MSE:
    mse_score = ridge_reg.score(dataset)
    print("MSE:", mse_score)

