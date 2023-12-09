#importação dos packages necessários:
from abc import abstractmethod
from typing import Union
import numpy as np
from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for activation layers.
    """

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: float) -> Union[float, np.ndarray]:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output error of the layer.
        """
        return self.derivative(self.input) * output_error #chama o método derivative da classe de ativação, passando a 
        #entrada (self.input) devolvendo a derivada da função de ativação aplicada à entrada que será  multiplicada pelo 
        #output_error, que é o erro de saída da camada

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Derivative of the activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The derivative of the activation function.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape #a forma de saída da camada é igual à forma de entrada

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0 #a camada não possui parâmetros treináveis, logo, o número de parameters é 0


class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.
    """
    def activation_function(self, input: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-input)) #recebe uma matriz de entrada input e calcula a saída aplicando a função sigmoide
        #a cada elemento da matriz
    

    def derivative(self, input: np.ndarray):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input) * (1 - self.activation_function(input)) 
        #esta fórmula é aplicada elemento a elemento na matriz de entrada. A derivada da função sigmoide é usada no 
        #algoritmo de backward propagation para calcular os gradientes durante o treino da rede neural


class ReLUActivation(ActivationLayer):
    """
    ReLU activation function.
    """
    def activation_function(self, input: np.ndarray):
        """
        ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.maximum(0, input) #devolve o máximo entre zero e cada elemento da matriz de entrada, garantindo que 
        #todos os valores negativos se tornem zero, utilizando o NumPy


    def derivative(self, input: np.ndarray):
        """
        Derivative of the ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return np.where(input > 0, 1, 0) #utiliza np.where para aplicar uma condição a todos os elementos da matriz input.
        #Se um elemento de input for maior que zero, a função devolve 1, caso contrário, devolve 0. Esta é a definição da 
        #derivada da função ReLU: é 1 para valores positivos e 0 para valores não positivos (zero ou negativos).
    

class TanhActivation(ActivationLayer):
    """
    Hyperbolic tangent (Tanh) activation function.
    """
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Tanh activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.tanh(input) #utiliza a função np.tanh(), que aplica a função hiperbólica tangente a cada elemento da 
        #matriz de entrada (input). Esta função devolve valores no intervalo de -1 a 1, comprimindo a entrada para um 
        #intervalo específico


    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Derivative of the tanh activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return 1 - np.tanh(input) ** 2 #utiliza a fórmula para a derivada da função Tanh, que é derivada do seu valor, 1 
        #menos o quadrado da função Tanh aplicada ao input. A expressão 1 - np.tanh(input) ** 2 calcula a derivada ponto 
        #a ponto para cada elemento na matriz de entrada (input)



class SoftmaxActivation(ActivationLayer):
    """
    Softmax activation function.
    """
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Softmax activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        #cálculo dos valores exponenciais dos elementos da matriz de entrada (input) subtraída pelo máximo valor ao longo 
        #do último eixo (axis=-1) para evitar problemas numéricos:
        exp_values = np.exp(input - np.max(input, axis=-1, keepdims=True))

        #divisão dos valores exponenciais pela soma ao longo do último eixo para normalizar os valores, resultando na saída 
        #da função Softmax:
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Derivative of the softmax activation function.

        This derivative is usually used during backpropagation in combination with the cross-entropy loss.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        softmax_output = self.activation_function(input) #cálculo da saída do Softmax para o input dado
        return softmax_output * (1 - softmax_output) #cálculo da derivada (é o produto entre a saída do Softmax e a 
        #diferença entre a saída do Softmax e 1)


#Testes:
if __name__ == '__main__':
    #Testes para SigmoidActivation
    sigmoid = SigmoidActivation()

    test_input = np.array([0.5, 0.2, 0.8])
    print("Sigmoid Activation - Test Input:", test_input)
    print("Sigmoid Activation - Output:", sigmoid.activation_function(test_input))
    print("Sigmoid Activation - Derivative:", sigmoid.derivative(test_input))

    #Testes para ReLUActivation
    relu = ReLUActivation()

    test_input = np.array([-1, 0, 2])
    print("\nReLU Activation - Test Input:", test_input)
    print("ReLU Activation - Output:", relu.activation_function(test_input))
    print("ReLU Activation - Derivative:", relu.derivative(test_input))

    #Testes para TanhActivation
    tanh = TanhActivation()

    test_input = np.array([-0.5, 0, 0.5])
    print("\nTanh Activation - Test Input:", test_input)
    print("Tanh Activation - Output:", tanh.activation_function(test_input))
    print("Tanh Activation - Derivative:", tanh.derivative(test_input))

    #Testes para SoftmaxActivation
    softmax = SoftmaxActivation()

    test_input = np.array([[1, 2, 3], [2, 4, 6]])
    print("\nSoftmax Activation - Test Input:", test_input)
    print("Softmax Activation - Output:", softmax.activation_function(test_input))
    print("Softmax Activation - Derivative:", softmax.derivative(test_input))
