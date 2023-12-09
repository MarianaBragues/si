#importação dos packages necessários:
import copy
from abc import abstractmethod
import numpy as np
from si.neural_networks.optimizers import Optimizer


class Layer:
    """
    Base class for neural network layers.
    """

    @abstractmethod
    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input, i.e., computes the output of a layer for a given input.

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
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error: float) -> float:
        """
        Perform backward propagation on the given output error, i.e., computes dE/dX for a given dE/dY and update
        parameters if any.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        raise NotImplementedError

    def layer_name(self) -> str:
        """
        Returns the name of the layer.

        Returns
        -------
        str
            The name of the layer.
        """
        return self.__class__.__name__

    @abstractmethod
    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        raise NotImplementedError

    def set_input_shape(self, shape: tuple):
        """
        Sets the shape of the input to the layer.

        Parameters
        ----------
        shape: tuple
            The shape of the input to the layer.
        """
        self._input_shape = shape

    def input_shape(self) -> tuple:
        """
        Returns the shape of the input to the layer.

        Returns
        -------
        tuple
            The shape of the input to the layer.
        """
        return self._input_shape

    @abstractmethod
    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        raise NotImplementedError


class DenseLayer(Layer):
    """
    Dense layer of a neural network.
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        #define o construtor da classe
        """
        Initialize the dense layer.

        Parameters
        ----------
        n_units: int
            The number of units of the layer, aka the number of neurons, aka the dimensionality of the output space.
        input_shape: tuple
            The shape of the input to the layer.
        """
        super().__init__() #chama o construtor da classe Layer
        self.n_units = n_units
        self._input_shape = input_shape
        #iniciados como None:
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None


    def initialize(self, optimizer: Optimizer) -> 'DenseLayer':
        #inicia os pesos da camada a partir de uma distribuição uniforme centrada em zero entre [-0.5, 0.5]:
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5 #cria uma matriz de forma (input_shape[0], n_units) 
        #com valores aleatórios entre 0 e 1. - 0.5 ajusta essa matriz aleatória para ter valores entre -0.5 e 0.5
        self.biases = np.zeros((1, self.n_units)) #inicia os vieses da camada como uma matriz de zeros com forma (1, n_units)
        
        #criam cópias profundas do otimizador fornecido: copy.deepcopy() cria uma cópia independente do optimizer
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)

        return self


    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape) #calcula o número total de parâmetros (a soma do 
        #número de elementos na matriz de pesos e na matriz de vieses)


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
        self.input = input #salva a entrada para a camada para uso posterior
        self.output = np.dot(self.input, self.weights) + self.biases #calcula a saída da camada
        #np.dot() realiza a multiplicação entre a entrada e os pesos da camada com recurso à função dot do NumPy
        #self.weights são os pesos da camada
        #self.biases são os vieses da camada
        return self.output #devplve a saída produzida pela camada após a multiplicação da entrada pelos pesos, com a adição dos vieses


    def backward_propagation(self, output_error: np.ndarray) -> float:
        """
        Perform backward propagation on the given output error.
        Computes the dE/dW, dE/dB for a given output_error=dE/dY.
        Returns input_error=dE/dX to feed the previous layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        #input_error: calcula o erro de entrada para a camada:
        input_error = np.dot(output_error, self.weights.T)
        #output_error: calcula o gradiente de erro em relação aos pesos:
        weights_error = np.dot(self.input.T, output_error)
        #bias_error: calcula o gradiente de erro em relação aos vieses:
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        #atualização dos pesos e vieses através de otimizador (self.w_opt e self.b_opt) com base nos gradientes de erro 
        #calculados:
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error #retorna o gradiente de erro em relação à entrada da camada. Esse valor será usado como o 
        #gradiente de erro da camada anterior durante o processo de backward_propagation


    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return (self.n_units,) #devolve a forma da saída da camada: tuple com um único valor, self.n_units, que representa
        #o número de unidades (neurónios) na camada (dimensão da saída da camada densa)
    

class Dropout(Layer):
    """
    A dropout layer in NNs is a regularization technique where a random set of neurons is temporarily ignored (dropped out)
    during training, helping prevent overfitting by promoting robustness and generalization in the model
    """

    def __init__(self, probability: float):
        #define o construtor da classe
        """
        Initialize the dropout layer.

        Parameters
        ----------
        probability: float
            the dropout rate, between 0 and 1;
        """
        super().__init__() #chama o construtor da classe Layer
        self.probability = probability
        self.mask = None #mask é iniciado como None (usado para ocultar neurónios aleatoriamente durante o treino)


    def forward_propagation(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        input: array
            input_array

        training: bool
            boolean of whether we are in training or inference mode
        """
        if training: #verifica se o modelo está em modo de treino
            scaling_factor = 1 / (1 - self.probability) #se sim, cria a variável scaling_factor que contém o cálculo de um
            #fator de escala com base na probabilidade de dropout. Este fator é usado para compensar a redução esperada na 
            #saída durante o treino devido ao dropout
            self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape) #gera uma máscara aleatória binomial
            #(com valores 0 ou 1) do mesmo formato que o input. Esta máscara determina quais neurénios serão desligados 
            #durante a fase de treino
            return input * self.mask * scaling_factor #se estiver em modo de treino, aplica a máscara ao input e aplica o 
            #fator de escala para corrigir o valor resultante. Realiza, assim, o dropout ao multiplicar os valores de 
            #entrada pelos valores da máscara (0 ou 1), removendo temporariamente alguns neurónios
        else: #se não estiver em modo de treino:
            return input #devolve o input sem alterações
        

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        output_error:  array
            the output error of the layer
        """
        return output_error * self.mask #é multiplicado o output_error pela máscara self.mask para desativar os mesmos 
        #neurónios que foram desativados durante a forward propagation na fase de treino. O objetivo do dropout é manter a
        #consistência entre os neurónios ativados e desativados entre as forward propagation e backward propagation durante
        #o treino, para evitar o overfitting


    def output_shape(self, input_shape: tuple) -> tuple:
        return input_shape #a forma de saída da camada Dropout é igual à forma de entrada

    def parameters(self) -> int:
        return 0  #a camada Dropout não possui parâmetros treináveis, logo, o número de parameters é 0
    

#Testes:
if __name__ == '__main__':
    #Testes para DenseLayer
    dense_layer = DenseLayer(n_units=5, input_shape=(10,))
    dense_layer.initialize(Optimizer(learning_rate=0.01))  

    print("Dense Layer - Output Shape:", dense_layer.output_shape())
    print("Dense Layer - Number of Parameters:", dense_layer.parameters())

    #criar um input de exemplo
    input_example = np.random.rand(100, 10)

    #Testa forward propagation na Dense Layer
    output_dense = dense_layer.forward_propagation(input_example, training=True)
    print("Dense Layer - Forward Propagation Output Shape:", output_dense.shape)

    #Testes para Dropout
    dropout_layer = Dropout(probability=0.5)

    #Teste forward propagation Dropout Layer
    output_dropout = dropout_layer.forward_propagation(output_dense, training=True)
    print("Dropout Layer - Forward Propagation Output Shape:", output_dropout.shape)

    # este backward propagation no Dropout Layer
    dropout_error_example = np.random.rand(100, 5)  #Exemplo de erro de saída
    dropout_input_error = dropout_layer.backward_propagation(dropout_error_example)
    print("Dropout Layer - Backward Propagation Input Error Shape:", dropout_input_error.shape)


