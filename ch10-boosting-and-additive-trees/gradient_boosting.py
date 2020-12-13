"""Implements the gradient boosting algorithm."""
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.tree
from tqdm import tqdm


def make_train_data(
    mean: np.ndarray,
    covariance: np.ndarray,
    num_data_points=2000,
) -> pd.DataFrame:
    """Randomly samples num_data_points 2D data points between [-3, 3] as
    features, and their labels are calculated as the density of 2D normal
    distribution.
    """
    df = pd.DataFrame(
        np.random.random(size=(num_data_points, 2)) * 6 - 3,
        columns=["x1", "x2"],
    )

    rv = scipy.stats.multivariate_normal(mean, covariance)
    noise = np.random.normal(loc=0, scale=10, size=len(df))
    return df.assign(label=rv.pdf(df))


def make_test_data(num_data_points=2000) -> pd.DataFrame:
    """Randomly samples num_data_points 2D data points between [-3, 3] as
    features, and their labels are calculated as the density of 2D normal
    distribution.
    """
    x1s_test, x2s_test = np.mgrid[
        -3:3:0.05,
        -3:3:0.05,
    ]

    X_test = np.concatenate(
        [
            x1s_test.reshape(-1, 1),
            x2s_test.reshape(-1, 1),
        ],
        axis=1,
    )


def visualize_training_data(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
) -> None:
    """Visualizes training data with 3D scattering plot."""
    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(1, 2, 1, projection="3d")

    ax.scatter(
        df[x_col],
        df[y_col],
        df[z_col],
        edgecolor="none",
        marker=".",
    )
    ax.set(xlabel=x_col, ylabel=y_col, zlabel=z_col)


class GradientBoostingRegressor:
    def __init__(
        self,
        learning_rate: float,
        n_estimators: int,
        max_depth: int,
    ) -> None:
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._n_estimators = n_estimators

        self._reset()

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def n_estimators(self) -> int:
        return self._n_estimators

    def _reset(self) -> None:
        self.estimators: List[sklearn.tree.DecisionTreeRegressor] = []
        self.loss_history: List[float] = []
        self.loss_val_history: List[float] = []

    def fit(self, X, y, X_val=None, y_val=None, verbose=True) -> None:
        """Fits a gradient boosting regressor with MSE loss function.

        NOTE: validation related variables are suffixed with _val.
        """
        self._reset()
        do_validation = X_val is not None and y_val is not None

        iterator = range(self.n_estimators)
        if verbose:
            iterator = tqdm(iterator)

        residuals = y
        preds = np.zeros(len(X))
        if do_validation:
            preds_val = np.zeros(len(X_val))

        for k in iterator:
            estimator = sklearn.tree.DecisionTreeRegressor(max_depth=self.max_depth)
            estimator.fit(X, residuals)
            self.estimators.append(estimator)
            preds += self._learning_rate * estimator.predict(X)
            residuals = y - preds
            loss = (residuals ** 2).mean()
            self.loss_history.append(loss)

            if do_validation:
                preds_val += self._learning_rate * estimator.predict(X_val)
                loss_val = np.mean((y_val - preds_val) ** 2)
                self.loss_val_history.append(loss_val)

    def predict(self, X, verbose=True) -> np.ndarray:
        preds = None
        estimators = tqdm(self.estimators) if verbose else self.estimators
        for estimator in estimators:
            if preds is None:
                preds = estimator.predict(X)
            else:
                preds += self._learning_rate * estimator.predict(X)
        return preds
