import abc
import numpy as np
import matplotlib.pyplot as plt

from intelligen.utils.validation import check_array
from ..metrics import mean_squared_error, r2_score

class LinearModel:

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit model."""
    
    def check_is_fitted(self):
        if not hasattr(self, '_fitted'): raise ValueError(
            f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    def predict(self, X=None):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,) or (n_targets, n_samples) (to test?)
            Returns predicted values.
        """
        self.check_is_fitted()
        if X is None: X = self.X
        else: X = check_array(X)
        return X @ self.coef_.T + self.intercept_
    
    def mse(self, y_true = None, y_pred = None, multioutput='uniform_average') -> float:
        """Returns the mean squared error
        Args:
            y_real (Vector, optional): Real data. Defaults to None (takes the fitted data)
            y_pred (Vector, optional): Predicted data. Defaults to None (takes the predicted data)
        Returns:
            float: Mean squared error
        """
        self.check_is_fitted()
        if y_true is None: y_true = self.y
        if y_pred is None: y_pred = self.predict()
        return mean_squared_error(y_true, y_pred, multioutput=multioutput)

    def score(self, y_true = None, y_pred = None, multioutput='uniform_average'):
        """Return the coefficient of determination of the prediction.
        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \\frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.
        """
        self.check_is_fitted()
        if y_true is None: y_true = self.y
        if y_pred is None: y_pred = self.predict()
        return r2_score(y_true, y_pred, multioutput=multioutput)

    def plot(self, ax=None, p_data = 1, n_data = 100) -> plt.Axes:
        """Plots the linear regression data against the real data
        Args:
            show (bool, optional): This shows the plot. Defaults to True.
            delimeters (bool, optional): This shows the delimeters of the surface that is plot. Defaults to False.
        """
        self.check_is_fitted()
        if self.y.ndim != 1: raise ValueError(f'multitarget is not supported at the moment')
        X = self.X
        y = self.y
        if p_data != 1:
            if not 0<=p_data<=1: raise ValueError(f'{p_data} is an invalid percentage')
            size = np.floor(X.shape[0]*p_data).astype(int)
            G = np.random.default_rng()
            index = G.choice(np.arange(len(y)), size=size, replace=False, shuffle=False)
            X, y = X[index], y[index]
        
        if len(y) > n_data:
            G = np.random.default_rng()
            index = G.choice(np.arange(len(y)), size=n_data, replace=False, shuffle=False)
            X, y = X[index], y[index]
        
        if self.n_features_in_ == 1:
            if ax is None: ax = plt.gca()
            ax.set_title('Simple Linear Regression')

            # min_y, max_y = np.min(self.y), np.max(self.y)
            # ax.set_ylim(min_y, max_y)
            min_x, max_x = np.min(self.X), np.max(self.X)
            # ax.set_xlim(min_x, max_x)
            min_y_pred, max_y_pred = self.predict(np.array([[min_x],[max_x]]))
            
            # ax.set_xmargin(10)
            # ax.set_ymargin(10)

            ax.set_xlabel('X')
            ax.set_ylabel('y')

            ax.plot((min_x, max_x), (min_y_pred, max_y_pred), c='red', label='Regression')
            ax.scatter(X, y, c='#325aa8', s=15, label='Data')
            # ax.legend()

        elif self.n_features_in_ == 2:
            if ax is None: ax = plt.gca(projection = '3d')
            ax.set_title('Multiple Linear Regression')

            min_x = np.min(self.X, axis = 0)
            max_x = np.max(self.X, axis = 0)

            x_axis = np.array([min_x[0],max_x[0]])
            y_axis = np.array([min_x[1],max_x[1]])

            x1, x2 = np.meshgrid(x_axis, y_axis)
            y_p = x1 * self.coef_[0] + x2 * self.coef_[1] + self.intercept_

            ax.plot_surface(x1, x2, y_p, color = 'royalblue', alpha = 0.5, label='Regression')
            ax.scatter(X[:, 0], X[:, 1], y, c = 'lightcoral', label='Data')
            # ax.legend()
        else:
            raise ValueError(f'plot doesn\'t support {self.num_features} number of features')
        return ax