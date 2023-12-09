#importação dos packages necessários:
from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, learning_rate: float):
        #define o construtor
        self.learning_rate = learning_rate #controlar a taxa de aprendizagem

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        #método abstrato update que deve ser implementado por todas as subclasses
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        #define o construtor
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate) #chama o construtor da classe Optimizer
        self.momentum = momentum 
        self.retained_gradient = None #inicia como None


    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None: #verifica se self.retained_gradient ainda não foi iniciado:
            self.retained_gradient = np.zeros(np.shape(w)) #se self.retained_gradient for None, é iniciado como um array 
            #de zeros com o mesmo formato que os pesos (w)

        #calculo do gradiente retido, que é uma combinação do gradiente anterior retido e o gradiente atual (grad_loss_w),
        #usando o valor de momentum:
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        #pesos são atualizados usando o gradiente retido e a taxa de aprendizagem:
        return w - self.learning_rate * self.retained_gradient #devolve a diferença entre os pesos atuais e a multiplicação
        #da taxa de aprendizagem pelo gradiente retido como os novos pesos da camada
    

class Adam(Optimizer):
    """
    Adam optimizer.
    """
    def __init__(self, learning_rate: float, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        #define o construtor
        """
        Initialize the Adam optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        beta_1: float
            The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
        beta_2: float
            The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
        epsilon: float
            A small constant for numerical stability. Defaults to 1e-8.
        """
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None #inicia como None
        self.v = None #inicia como None
        self.t = 0 #t é zero

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer using the Adam optimizer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.m is None: #verifica se m é None:
            self.m = np.zeros_like(w) #se m for None, ambos m como v são iniciados como arrays de zeros com o mesmo 
            #formato que os pesos (w). Estes arrays são usados para calcular os momentos das médias e variâncias dos gradientes
            self.v = np.zeros_like(w)

        self.t += 1 #incrementa o contador de tempo t para atualizar os momentos e calcular os estimadores não centrados 
        #das médias e variâncias

        #atualiza os valores de m e v:
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)

        #calcula estimativas não centradas das médias (m_hat) e variâncias (v_hat) corrigidas de bias:
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        #atualiza os pesos w usando a taxa de aprendizagem (learning_rate), os momentos das médias (m_hat) e variâncias 
        #(v_hat) e a constante epsilon para estabilidade numérica
        w -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return w #devolve os pesos atualizados da camada
    

#Testes:
if __name__ == '__main__':
    #Testes para a classe SGD
    sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)

    # Teste para a atualização dos pesos
    current_weights = np.array([0.5, 0.3, -0.2])
    grad_loss_weights = np.array([0.1, -0.2, 0.4])
    updated_weights_sgd = sgd_optimizer.update(current_weights, grad_loss_weights)
    print("Updated Weights (SGD):", updated_weights_sgd)


    #Testes para a classe Adam
    adam_optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # Teste para a atualização dos pesos
    current_weights = np.array([0.5, 0.3, -0.2])
    grad_loss_weights = np.array([0.1, -0.2, 0.4])
    updated_weights_adam = adam_optimizer.update(current_weights, grad_loss_weights)
    print("Updated Weights (Adam):", updated_weights_adam)
   

