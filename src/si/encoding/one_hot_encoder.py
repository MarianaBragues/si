#importação dos packages necessários:
from typing import List, Optional, Dict
import numpy as np


class OneHotEncoder:
    """
    One hot encoding is a representation technique where categorical data, such as words in a text sequence or characters 
    in a sequence), is converted into binary vectors with only one element set to 1 (indicating the presence of a 
    specific category) and the rest set to 0.
    """
    def __init__(self, padder: Optional[str] =None, max_length: Optional[int] =None):
        #definir o construtor
        """
        Parameters
        ----------
        padder: str
            character to perform padding with

        max_length: int
            maximum length of the sequences

        Estimated parameters
        --------------------
        alphabet: list
            the unique characters in the sequences

        char_to_index: dict
            dictionary mapping characters in the alphabet to unique integers

        index_to_char: dict
            reverse of char_to_index dictionary mapping integers to characters
        """
        self.padder: Optional[str] = padder
        self.max_length: Optional[int] = max_length
        self.alphabet: Optional[List[str]] = None
        self.char_to_index: Optional[Dict[str, int]] = None
        self.index_to_char: Optional[Dict[int, str]] = None


    def fit(self, data: list) -> None:
        """
        Fits the encoder to the data (learns the alphabet , char_to_index and index_to_char) + sets max_length if not 
        defined

        Parameters
        ----------
        data: list
            list of sequences (strings) to learn from
        """
        #cria o atributo alphabet, que contém os caracteres únicos presentes em todas as sequências fornecidas em data. 
        #set(''.join(data)) cria uma string única combinando todas as sequências em data, e sorted ordena os caracteres únicos:
        self.alphabet = sorted(set(''.join(data)))

        #cria um dicionário char_to_index que mapeia cada caracter do alphabet para um índice único. 
        #enumerate gera pares índice-caracteres para cada caracter no alphabet:
        self.char_to_index = {char: index for index, char in enumerate(self.alphabet)}

        #cria um dicionário index_to_char que mapeia cada índice único de volta para o seu caracter correspondente no 
        #alphabet:
        self.index_to_char = {index: char for index, char in enumerate(self.alphabet)}

        if self.max_length is None: #verifica se max_length não está definido
            self.max_length = max(len(seq) for seq in data) #se max_length não estiver definido, determina o valor máximo
            #de comprimento das sequências em data e atribui esse valor a max_length


    def transform(self, data: List[str]) -> List[np.ndarray]:
        """
        Encodes the sequence to one hot encoding

        Parameters
        ----------
        data: list 
            data to encode
        """
        encoded_sequences = [] #inici uma lista vazia para armazenar as sequências codificadas - encoded_sequences
        for sequence in data: #itera sobre cada sequência em data:
            if len(sequence) > self.max_length: #verifica se o comprimento da sequência atual é maior que max_length
                sequence = sequence[:self.max_length] #se a sequência for maior que max_length, é cortada para o comprimento máximo
            elif len(sequence) < self.max_length: #verifica se o comprimento da sequência é menor que max_length
                sequence = sequence + (self.padder * (self.max_length - len(sequence))) #se a sequência for menor que 
                #max_length, é preenchida com padder até atingir o comprimento máximo

            encoded_sequence = np.zeros((self.max_length, len(self.alphabet))) #inicializa uma matriz de zeros para 
            #armazenar a representação one-hot da sequência. A matriz tem o tamanho (max_length, len(alphabet))
            for i, char in enumerate(sequence): #itera sobre cada caracter na sequência:
                if char in self.char_to_index: #verifica se o caracter está presente no dicionário char_to_index
                    encoded_sequence[i, self.char_to_index[char]] = 1 #define o valor correspondente ao índice do 
                    #caractere como 1
            encoded_sequences.append(encoded_sequence) #adiciona a representação one-hot da sequência à lista de sequências codificadas

        return encoded_sequences #devolve a lista de sequências codificadas em representações one-hot
    

    def fit_transform(self, data: List[str]) -> List[np.ndarray]:
        """
        Runs fit and the transform
        
        Parameters
        ----------
        data: List[str]
            list of sequences (strings) to learn from
        """
        self.fit(data) #executa a etapa de ajuste após chamar a função fit
        return self.transform(data) #devolve os dados transformados após chamar a função transform


    def inverse_transform(self, encoded_sequences: List[np.ndarray]) -> List[str]:
        """
        Convert the one hot encoded matrices back to sequences using the index_to_char dictionary.

        Parameters
        ----------
        encoded_sequences: List[np.ndarray]
            data to decode one hot encoded matrices
        """
        decoded_sequences = [] #inicia uma lista vazia para armazenar as sequências descodificadas
        for encoded_sequence in encoded_sequences: #itera sobre cada matriz codificada one-hot:
            #para cada matriz, encontra o índice do valor máximo em cada linha (axis=1), obtendo assim o índice de cada 
            #caracter e usa esses índices para obter o caracter correspondente do dicionário index_to_char
            #junta os caracteres para formar a sequência descodificada:
            decoded_sequence = ''.join([self.index_to_char[index] for index in encoded_sequence.argmax(axis=1)])
            decoded_sequences.append(decoded_sequence) #adiciona a sequência descodificada à lista

        return decoded_sequences #devolve a lista de sequências descodificadas
    
#Testes:
if __name__ == "__main__":
    #criar uma instância da classe OneHotEncoder
    encoder = OneHotEncoder()

    #dados de exemplo
    data = ["abc", "def", "ghi"]

    #Testa o método fit
    print("Fit data:")
    encoder.fit(data)
    print("Alphabet:", encoder.alphabet)
    print("Char to Index:", encoder.char_to_index)
    print("Index to Char:", encoder.index_to_char)

    #Testa o método transform
    print("\nTransformed data:")
    encoded_data = encoder.transform(data)
    for idx, seq in enumerate(encoded_data):
        print(f"Sequence {idx + 1}:", seq)

    #Testa o método inverse_transform
    print("\nInverse Transformed data:")
    decoded_data = encoder.inverse_transform(encoded_data)
    for idx, seq in enumerate(decoded_data):
        print(f"Decoded Sequence {idx + 1}:", seq)

    #Verifica se os dados originais são iguais aos dados decodificados
    assert data == decoded_data, "Erro: Os dados decodificados não correspondem aos dados originais."
