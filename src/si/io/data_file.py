#importação dos packages necessários:
import numpy as np
from si.data.dataset import Dataset


def read_data_file(filename: str,
                   sep: str = None,
                   label: bool = False) -> Dataset:
    """
    Reads a data file into a Dataset object

    Parameters
    ----------
    filename : str
        Path to the file
    sep : str, optional
        The separator used in the file, by default None
    label : bool, optional
        Whether the file has a label, by default False

    Returns
    -------
    Dataset
        The dataset object
    """
    raw_data = np.genfromtxt(filename, delimiter=sep) #lê o arquivo especificado por filename usando a função genfromtxt 
    #do NumPy para carregar os dados. Usa delimiter=sep para definir o separador dos dados no arquivo

    if label: #verifica se a label é verdadeira:
        X = raw_data[:, :-1] #se a label for verdadeira, atribui a X todas as colunas do raw_data, exceto a última
        y = raw_data[:, -1] #se a label for verdadeira, atribui a y a última coluna do raw_data

    else: #se a label for falsa:
        X = raw_data #atribui a X todos os dados 
        y = None #y é None

    return Dataset(X, y) #devolve o dataset com os dados X e as labels y


def write_data_file(filename: str,
                    dataset: Dataset,
                    sep: str = None,
                    label: bool = False) -> None:
    """
    Writes a Dataset object to a data file

    Parameters
    ----------
    filename : str
        Path to the file
    dataset : Dataset
        The dataset object
    sep : str, optional
        The separator used in the file, by default None
    label : bool, optional
        Whether to write the file with label, by default False
    """
    if not sep: #verifica se sep é falso
        sep = " " #define sep como um espaço em branco

    if label: #se a label é verdadeira
        data = np.hstack((dataset.X, dataset.y.reshape(-1, 1))) #se label for verdadeira, concatena as colunas do dataset.X
        #com dataset.y, que é redimensionado para se tornar numa matriz de uma coluna. Esta concatenação ocorre 
        #horizontalmente usando a função hstack do NumPy
    else: #se a label é falsa
        data = dataset.X #define data como os dados do dataset.X

    return np.savetxt(filename, data, delimiter=sep) #usa a função savetxt do NumPy para guardar os dados data no arquivo 
    #especificado por filename, usando o separador sep
