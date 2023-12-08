import numpy as np #importação do NumPy

#define a função manhattan_distance, que calcula a distância de Manhattan entre um ponto x e um conjunto de pontos y.
#x e y são matrizes NumPy (np.ndarray)

def manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates the Manhattan distance between X and Y using the following formula:
        distance_x_y1 = |x1 y11| + |x2 y12| + ... + | xn y1n|
        distance_x_y2 =|x1 y21| + |x2 y22| + ... + | xn y2n|

    Parameters
    ----------
    x: np.ndarray
        a single sample
    y: np.ndarray
        multiple samples

    Returns
    -------
    np.ndarray
        an array containing the distances between X and the various samples in Y.
    """
    distances = [] #cria uma lista vazia "distances"
    
    for sample in y: #para cada amostra em y:
        distance = sum(abs(xi - yi) for xi, yi in zip(x, sample)) #cria a variável distance onde calcula a distância de 
        #Manhattan entre x e a amostra atual percorrendo coordenada por coordenada (zip(x, sample)) e somando as diferenças
        #absolutas.
        distances.append(distance) #adiciona os resultados calculados anteriores 
    
    return distances #retorna uma matriz NumPy que contém as distâncias entre x e as várias amostras em y


#Testes:
if __name__ == '__main__':
    # test manhattan_distance
    x = np.array([1, 2, 3])
    y = np.array([[4, 5, 6], [7, 8, 9]])
    our_distance = manhattan_distance(x, y)
   