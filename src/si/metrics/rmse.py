import math

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
    if len(y_true) != len(Y_pred):
        raise ValueError("Input lists must have the same length")
    # Calculate the squared differences between y_true and y_pred
    squared_diff = [(true - pred) ** 2 for true, pred in zip(y_true, Y_pred)]
    # Calculate the mean of squared differences
    mean_squared_diff = sum(squared_diff) / len(y_true)
    # Calculate the square root of the mean squared difference
    rmse = math.sqrt(mean_squared_diff)

    return rmse