#importação do NumPy necessário à função:
import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the accuracy of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    accuracy: float
        The accuracy of the model
    """
    # deal with predictions like [[0.52], [0.91], ...] and [[0.3, 0.7], [0.6, 0.4], ...]
    # they need to be in the same format: [0, 1, ...] and [1, 0, ...]
    #define uma função interna correct_format que trata a formatação das labels preditos ou verdadeiros para garantir que 
    #estejam no formato correto para comparação:
    def correct_format(y):
        if len(y[0]) == 1: #verifica se cada item no array y é uma lista de comprimento 1:
            corrected_y = [np.round(y[i][0]) for i in range(len(y))] #se sim, arredonda cada valor para o inteiro mais 
            #próximo, convertendo os valores de probabilidade para valores discretos
        else: #se não: 
            corrected_y = [np.argmax(y[i]) for i in range(len(y))] #determina o índice do valor máximo em cada lista de y
        return np.array(corrected_y) #devolve um array NumPy com as labels no formato correto para comparação (discreto ou arredondado) 
    if isinstance(y_true[0], list) or isinstance(y_true[0], np.ndarray): #verifica se o primeiro elemento de y_true é uma lista ou um array NumPy
        y_true = correct_format(y_true) #se sim, chama a função correct_format para ajustar o formato de y_true
    if isinstance(y_pred[0], list) or isinstance(y_pred[0], np.ndarray): #verifica se o primeiro elemento de y_pred é uma lista ou um array NumPy
        y_pred = correct_format(y_pred) #se sim, chama a função correct_format para ajustar o formato de y_pred
    return np.sum(y_pred == y_true) / len(y_true) #conta o número de predições corretas. O resultado é dividido pelo 
    #número total de instâncias (len(y_true)) para calcular a proporção de predições corretas em relação ao total de 
    #instâncias, resultando na precisão do modelo