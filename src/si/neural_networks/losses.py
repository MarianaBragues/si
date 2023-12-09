#importação dos packages necessários:
from abc import abstractmethod
import numpy as np


class LossFunction:

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the derivative of the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean squared error loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        #cálculo do erro quadrático médio entre as labels verdadeiras (y_true) e as labels preditos (y_pred). Subtrai os 
        #dois arrays, eleva ao quadrado cada diferença e calcula a média desses quadrados. O resultado é o erro quadrático
        #médio entre y_true e y_pred, que é devolvido como um número float
        return np.mean((y_true - y_pred) ** 2)


    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        #calculo da derivada da função de erro quadrático médio:
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Cross entropy loss function.
    """
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        #a variável p é definida como y_pred, mas com valores ajustados usando a função clip do NumPy para evitar valores 
        #muito pequenos (próximos a zero) ou muito grandes (próximos a um), assim, evita-se problemas de logaritmos 
        #próximos a zero ou indefinidos:
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)) #calcula a entropia cruzada binária usando a 
        #a fórmula matemática respetiva entre os rótulos verdadeiros (y_true) e os rótulos preditos ajustados (p) 


    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        #a variável p é definida como y_pred, mas com valores ajustados usando a função clip do NumPy para evitar valores 
        #muito pequenos (próximos a zero) ou muito grandes (próximos a um)
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return - (y_true / p) + (1 - y_true) / (1 - p) #calculo da derivada da função de perda de entropia cruzada binária
    


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical cross-entropy loss function.
    """
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the categorical cross-entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels in one-hot encoded format.
        y_pred: numpy.ndarray
            The predicted probabilities for each class.

        Returns
        -------
        float
            The loss value.
        """
        #os valores de y_pred são ajustados usando a função clip do NumPy para evitar valores muito pequenos (próximos a 
        #zero) ou muito grandes (próximos a um)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -np.sum(y_true * np.log(y_pred)) #calculo da entropia cruzada categórica


    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the categorical cross-entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels in one-hot encoded format.
        y_pred: numpy.ndarray
            The predicted probabilities for each class.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        #os valores de y_pred são ajustados usando a função clip do NumPy para evitar valores muito pequenos (próximos a 
        #zero) ou muito grandes (próximos a um)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return - (y_true / y_pred) #calculo da derivada da função de perda de entropia cruzada categórica
    

#Testes:
if __name__ == '__main__':
    from si.metrics import mse
    #Teste para a classe MeanSquaredError
    mse = MeanSquaredError()

    #Teste para o cálculo do erro
    y_true_mse = np.array([1, 2, 3])
    y_pred_mse = np.array([1.5, 2.2, 2.8])
    mse_value = mse.loss(y_true_mse, y_pred_mse)
    print("Mean Squared Error:", mse_value)

    #Teste para o cálculo da derivada do erro
    mse_derivative = mse.derivative(y_true_mse, y_pred_mse)
    print("Mean Squared Error Derivative:", mse_derivative)


    #Testes para a classe BinaryCrossEntropy
    bce = BinaryCrossEntropy()

    #Teste para o cálculo do erro
    y_true_bce = np.array([0, 1, 1])
    y_pred_bce = np.array([0.2, 0.8, 0.9])
    bce_value = bce.loss(y_true_bce, y_pred_bce)
    print("Binary Cross Entropy:", bce_value)

    #Teste para o cálculo da derivada do erro
    bce_derivative = bce.derivative(y_true_bce, y_pred_bce)
    print("Binary Cross Entropy Derivative:", bce_derivative)


    #Testes para a classe CategoricalCrossEntropy
    cce = CategoricalCrossEntropy()

    #Teste para o cálculo do erro
    y_true_cce = np.array([[0, 1, 0], [1, 0, 0]])
    y_pred_cce = np.array([[0.1, 0.9, 0.4], [0.8, 0.1, 0.1]])
    cce_value = cce.loss(y_true_cce, y_pred_cce)
    print("Categorical Cross Entropy:", cce_value)

    #Teste para o cálculo da derivada do erro
    cce_derivative = cce.derivative(y_true_cce, y_pred_cce)
    print("Categorical Cross Entropy Derivative:", cce_derivative)
