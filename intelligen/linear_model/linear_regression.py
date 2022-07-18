import numpy as np
import matplotlib.pyplot as plt
from ..metrics import mean_squared_error
from ..utils.validation import _num_features, check_X_y
from .ABCLinearModel import LinearModel
from scipy.optimize import nnls


# Bibliography https://github.com/arseniyturin/SGD-From-Scratch/blob/master/Gradient%20Descent.ipynb

class LinearRegression(LinearModel):
    """
    Ordinary least squares Linear Regression
    ========================================

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.
        This uses always `scipy.optimize.nnls` to fit the model
    implementation : str, default='numpy'
        The implementation followed to fit the model:
            'mine': if n_features is:
                1: `(mean(X)*mean(y)-mean(X*y))/(mean(x)**2-mean(x**2))`
                esle: `(X.T @ X)^-1 @ X.T @ y`
            'numpy': This uses `numpy.linalg.lstsq`
        Recommended to use the default 'numpy', note that 'mine' has an inverse,
        so for large n_features, it's very slow, but for low n_features and large
        n_samples, it's actually faster

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.
    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    """
    def __init__(self, positive = False, implementation = 'numpy') -> None:
        self.positive = positive
        self.implementation = implementation

    def fit(self, X, y) -> None:
        """
        Fits the data and calculates the coefficients of the linear regression

        Parameters
        ----------
        X : Array-like, shape(n_samples, n_features)
            Data
        y : Array-like, shape(n_samples,) or (n_samples, n_targets)
            Target
        """
        
        # This only ensures X is 2D, y can be 1D
        self.X, self.y = check_X_y(X, y)
        self.n_features_in_ = _num_features(self.X)
        X, y = self.X, self.y

        if self.positive:
            # We add the coefficient column
            # nnls needs y to be 1D
            X = np.column_stack((np.ones(X.shape[0]), X))
            if y.ndim < 2:
                temp = nnls(X,y)[0]
                self.intercept_, self.coef_ = temp[0], temp[1:]
            else:
                # slicing a column y[:,j] gives 1D array
                temp = np.array([nnls(X, y[:,j])[0] for j in range(y.shape[1])])
                self.intercept_, self.coef_ = temp[:,0], temp[:,1:]
            
        else:
            if self.implementation == 'mine':
                # Simple linear regression
                if self.n_features_in_ == 1:
                    # Least Square Error (minimizes mean square error)
                    # y must be vertical, by default it is when there are more targets
                    if y.ndim < 2:
                        y = y.reshape(-1,1)
                    self.coef_ = (np.mean(X) * np.mean(y, axis=0) - np.mean(X*y, axis=0)) / ((np.mean(X)**2) - np.mean(X**2))
                    self.intercept_ = np.mean(y, axis=0) - self.coef_ * np.mean(X)
                    
                    # Consistent reshaping
                    if len(self.coef_) != 1: self.coef_ = np.array(self.coef_).reshape(-1,1)
                    if len(self.intercept_) == 1: self.intercept_ = self.intercept_[0]

                    # Alternative option, but actually slower:
                    # scipy.stats.linregress(X, y)

                # Multiple linear regression
                else:
                    # We add the coefficient column
                    X = np.column_stack((np.ones(X.shape[0]), X))
                    # We transpose to keep consistancy
                    temp = (np.linalg.pinv(X.T @ X) @ X.T @ y).T
                    if y.ndim < 2:
                        self.intercept_, self.coef_ = temp[0], temp[1:]
                    else:
                        self.intercept_, self.coef_= temp[:,0], temp[:,1:]
            
            elif self.implementation == 'numpy':
                # We add the coefficient column
                X = np.column_stack((np.ones(X.shape[0]), X))
                # We transpose to keep consistancy
                temp = np.linalg.lstsq(X, y, rcond=None)[0].T
                if y.ndim < 2:
                    self.intercept_, self.coef_ = temp[0], temp[1:]
                else:
                    self.intercept_, self.coef_= temp[:,0], temp[:,1:]
            else:
                raise ValueError(f'{self.implementation} not supported, try numpy or mine')

        self._fitted = True
        return self