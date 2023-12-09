#importação dos packages necessários:
import pandas as pd
from si.data.dataset import Dataset


def read_csv(filename: str,
             sep: str = ',',
             features: bool = False,
             label: bool = False) -> Dataset:
    """
    Reads a csv file (data file) into a Dataset object

    Parameters
    ----------
    filename : str
        Path to the file
    sep : str, optional
        The separator used in the file, by default ','
    features : bool, optional
        Whether the file has a header, by default False
    label : bool, optional
        Whether the file has a label, by default False

    Returns
    -------
    Dataset
        The dataset object
    """
    data = pd.read_csv(filename, sep=sep) #lê o arquivo CSV usando a biblioteca Pandas e armazena o conteúdo na variável 
    #data. O separador utilizado é o especificado por sep

    if features and label: #se features e label forem verdadeiras:
        features = data.columns[:-1] #define features como as colunas do DataFrame, excluindo a última coluna
        label = data.columns[-1] #define label como a última coluna do DataFrame
        X = data.iloc[:, :-1].to_numpy() #define X como uma matriz NumPy contendo todas as linhas e todas as colunas, exceto a última
        y = data.iloc[:, -1].to_numpy() #define y como um array NumPy contendo todas as linhas e apenas a última coluna

    elif features and not label: #se as features forem verdadeiras e as label falsas:
        features = data.columns #define features como todas as colunas do DataFrame
        X = data.to_numpy() #define X como uma matriz NumPy contendo todas as linhas e colunas do DataFrame
        y = None #y é None

    elif not features and label: #se as features forem falsas e label for verdadeira:
        X = data.iloc[:, :-1].to_numpy() #define X como uma matriz NumPy contendo todas as linhas e todas as colunas, exceto a última
        y = data.iloc[:, -1].to_numpy() #define y como uma matriz NumPy contendo apenas a última coluna do DataFrame
        features = None #features é None
        label = None #label é None

    else: #se as features e as labels forem falsas:
        X = data.to_numpy() #define X como uma matriz NumPy contendo todos os dados do Dataframe
        y = None #y é None 
        features = None #features é None
        label = None #label é None

    return Dataset(X, y, features=features, label=label) #devolve o dataset


def write_csv(filename: str,
              dataset: Dataset,
              sep: str = ',',
              features: bool = False,
              label: bool = False) -> None:
    """
    Writes a Dataset object to a csv file

    Parameters
    ----------
    filename : str
        Path to the file
    dataset : Dataset
        The dataset object
    sep : str, optional
        The separator used in the file, by default ','
    features : bool, optional
        Whether the file has a header, by default False
    label : bool, optional
        Whether the file has a label, by default False
    """
    data = pd.DataFrame(dataset.X) #cria um DataFrame do pandas a partir do atributo X do objeto Dataset

    if features: #se as features forem verdadeiras:
        data.columns = dataset.features #atribui os nomes das colunas do DataFrame (data.columns) para os nomes dos 
        #do Dataset (dataset.features)

    if label: #se a label for verdadeira:
        data[dataset.label] = dataset.y #adiciona uma coluna ao DataFrame chamada de acordo com a label do Dataset e 
        #atribui a essa coluna os valores do atributo y do Dataset

    data.to_csv(filename, sep=sep, index=False) #escreve o DataFrame no arquivo CSV com o nome especificado (filename), 
    #usando o separador definido (sep). O parâmetro index=False impede que o índice do DataFrame seja incluído no arquivo CSV
