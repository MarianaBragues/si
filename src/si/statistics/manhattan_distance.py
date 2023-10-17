import numpy as np


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
    distances = []
    
    for sample in y:
        distance = sum(abs(xi - yi) for xi, yi in zip(x, sample))
        distances.append(distance)
    
    return distances


if __name__ == '__main__':
    # test manhattan_distance
    x = np.array([1, 2, 3])
    y = np.array([[4, 5, 6], [7, 8, 9]])
    our_distance = manhattan_distance(x, y)
   