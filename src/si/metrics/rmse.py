import math #importação da biblioteca math

def rmse(y_true: float, Y_pred: float) -> float:
    """
    Calculates the error following the RMSE formula

    Parameters
    ----------
    y_true: 
        real values of y
    Y_pred: 
        predicted values of y

    Returns: float
        float corresponding to the error between y_true and y_pred

    """
    #esta função recebe duas listas (y_true e Y_pred) que contêm os valores reais e preditos, respectivamente
    if len(y_true) != len(Y_pred): #verifica se as duas listas têm o mesmo comprimento
        raise ValueError("Input lists must have the same length") #Se não tiverem, cria um ValueError
    
    if not y_true or not Y_pred: #verifica se as listas estão vazias
        return 0.0  #devolve 0 se uma das listas estiver vazia

    squared_diff = [(true - pred) ** 2 for true, pred in zip(y_true, Y_pred)] #cria a variável squared_diff que guarda os 
    #valores do cálculo das diferenças ao quadrado entre cada valor real (y_true) e o seu respectivo valor predito (Y_pred)
    
    mean_squared_diff = sum(squared_diff) / len(y_true) #cria a variável mean_squared_diff que guarda os valores do cálculo
    #da média das diferenças ao quadrado
 
    rmse = math.sqrt(mean_squared_diff) #cria a variável rmse que guarda os valores do cálculo da raiz quadrada da média 
    #das diferenças ao quadrado

    return rmse #devolve o resultado do RMSE

#Testes:
if __name__ == "__main__":
    #Teste 1: Valores idênticos, RMSE deve ser 0
    y_true_1 = [3, -0.5, 2, 7]
    y_pred_1 = [3, -0.5, 2, 7]
    assert rmse(y_true_1, y_pred_1) == 0, "Erro no Caso de Teste 1"

    #Teste 2: Valores diferentes
    y_true_2 = [3, -0.5, 2, 7]
    y_pred_2 = [2.5, 0.0, 2, 8]
    assert math.isclose(rmse(y_true_2, y_pred_2), 0.612, rel_tol=1e-3), "Erro no Caso de Teste 2"

    #Teste 3: Listas vazias, RMSE deve ser 0
    y_true_3 = []
    y_pred_3 = []
    assert rmse(y_true_3, y_pred_3) == 0, "Erro no Caso de Teste 3"
