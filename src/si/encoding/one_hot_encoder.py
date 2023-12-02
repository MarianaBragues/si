from typing import List, Optional, Dict
import numpy as np


class OneHotEncoder:
    """
    One hot encoding is a representation technique where categorical data, such as words in a text sequence or characters 
    in a sequence), is converted into binary vectors with only one element set to 1 (indicating the presence of a 
    specific category) and the rest set to 0.
    """
    def __init__(self, padder: Optional[str] =None, max_length: Optional[int] =None):
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
        self.alphabet = sorted(set(''.join(data)))
        self.char_to_index = {char: index for index, char in enumerate(self.alphabet)}
        self.index_to_char = {index: char for index, char in enumerate(self.alphabet)}

        # Set max_length if not defined
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in data)


    def transform(self, data: List[str]) -> List[np.ndarray]:
        """
        Encodes the sequence to one hot encoding

        Parameters
        ----------
        data: list 
            data to encode
        """
        encoded_sequences = []
        for sequence in data:
            # Trim or pad sequences to max_length
            if len(sequence) > self.max_length:
                sequence = sequence[:self.max_length]
            elif len(sequence) < self.max_length:
                sequence = sequence + (self.padder * (self.max_length - len(sequence)))

            # Encode the sequence to one-hot encoding
            encoded_sequence = np.zeros((self.max_length, len(self.alphabet)))
            for i, char in enumerate(sequence):
                if char in self.char_to_index:
                    encoded_sequence[i, self.char_to_index[char]] = 1
            encoded_sequences.append(encoded_sequence)

        return encoded_sequences
    

    def fit_transform(self, data: List[str]) -> List[np.ndarray]:
        """
        Runs fit and the transform
        
        Parameters
        ----------
        data: List[str]
            list of sequences (strings) to learn from
        """
        self.fit(data)
        return self.transform(data)


    def inverse_transform(self, encoded_sequences: List[np.ndarray]) -> List[str]:
        """
        Convert the one hot encoded matrices back to sequences using the index_to_char dictionary.

        Parameters
        ----------
        encoded_sequences: List[np.ndarray]
            data to decode one hot encoded matrices
        """
        decoded_sequences = []
        for encoded_sequence in encoded_sequences:
            decoded_sequence = ''.join([self.index_to_char[index] for index in encoded_sequence.argmax(axis=1)])
            decoded_sequences.append(decoded_sequence)
        return decoded_sequences
