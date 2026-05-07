import abc

import numpy as np

from intelligen.metrics import mean_squared_error, r2_score
from intelligen.utils.plot_utils import (
    plot,
    plot_surface,
    scatter,
    scatter_3d,
    set_labels,
)
from intelligen.utils.types import PlotReturnType


class LinearModel:
    """Abstract base class for linear models."""

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

        return X @ self.coef_.T + self.intercept_

    def mse(self, y_true = None, y_pred = None, multioutput='uniform_average') -> float:
        """Return the mean squared error.

        Parameters
        ----------
            y_real (Vector, optional): Real data. Defaults to None (takes the fitted data).
            y_pred (Vector, optional): Predicted data. Defaults to None (takes the predicted data).

        Returns
        -------
            float: Mean squared error.
        """
        self.check_is_fitted()
        if y_true is None: y_true = self.y
        if y_pred is None: y_pred = self.predict()
        return mean_squared_error(y_true, y_pred, multioutput=multioutput)

    def score(self, y_true = None, y_pred = None, multioutput='uniform_average'):
        """Return the coefficient of determination of the prediction.

        Parameters
        ----------
        y_true : array-like, shape (n_samples,) or (n_samples, n_targets)
            True target values.

        Returns
        -------
        float
            The coefficient of determination R^2 of the prediction.
        """
        self.check_is_fitted()
        if y_true is None: y_true = self.y
        if y_pred is None: y_pred = self.predict()
        return r2_score(y_true, y_pred, multioutput=multioutput)

    def plot(self, ax=None, p_data = 1, n_data = 100) -> PlotReturnType:
        """Plot the linear regression data against the real data.

        Parameters
        ----------
            ax (PlotReturnType, optional): The axes to plot on. Defaults to None (creates a new one).
            p_data (float, optional): Percentage of data to use for plotting. Defaults to 1 (100%).
            n_data (int, optional): Number of data points to plot. Defaults to 100.

        Returns
        -------
            PlotReturnType: The axes with the plot.
        """
        self.check_is_fitted()
        if self.y.ndim != 1: raise ValueError('multitarget is not supported at the moment')
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
            min_x, max_x = np.min(self.X), np.max(self.X)
            min_y_pred, max_y_pred = self.predict(np.array([[min_x],[max_x]]))

            # 1. Plot the scatter data
            ax = scatter(X, y, ax=ax)

            # 2. Plot the regression line
            # Note: passing tuples/lists of X and Y coordinates
            ax = plot([min_x, max_x], [min_y_pred, max_y_pred], ax=ax)

            # 3. Apply the formatting
            ax = set_labels(
                title='Simple Linear Regression',
                xlabel='X',
                ylabel='y',
                ax=ax
            )
            return ax


        elif self.n_features_in_ == 2:
            min_x = np.min(self.X, axis=0)
            max_x = np.max(self.X, axis=0)

            x_axis = np.array([min_x[0], max_x[0]])
            y_axis = np.array([min_x[1], max_x[1]])

            x1, x2 = np.meshgrid(x_axis, y_axis)
            y_p = x1 * self.coef_[0] + x2 * self.coef_[1] + self.intercept_

            # Use your backend wrappers!
            ax = plot_surface(x1, x2, y_p, ax=ax, label='Regression')
            ax = scatter_3d(X[:, 0], X[:, 1], y, ax=ax, label='Data')
            ax = set_labels(title='Multiple Linear Regression', ax=ax)
        else:
            raise ValueError(f'plot doesn\'t support {self.n_features_in_} number of features')
        return ax
