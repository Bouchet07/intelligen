import numpy as np

__all__ = ['mean_squared_error', 'r2_score']

def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters
    ----------
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.

    Returns
    -------
    float: The Mean Squared Error.
    """
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """
    Calculate the R-squared (coefficient of determination) score.

    Parameters
    ----------
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.

    Returns
    -------
    float: The R-squared score.
    """
    u = np.sum((y_true - y_pred) ** 2)
    v = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - u / v if v != 0 else 0.0
