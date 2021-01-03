"""Implements the gradient boosting algorithm."""
from typing import List
from unittest.case import expectedFailure

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.tree
from sklearn.utils import all_estimators
from tqdm import tqdm


def _decision_boundary(x1, x2):
    """An arbitrary decision boundary function"""
    # return x1 + x2 - 1 > 0
    return (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2 > 0.1


def make_data(num=500) -> pd.DataFrame:
    """Makes data with labels.

    Args:
        num: number of data points to make.
    """
    data = np.random.rand(num, 2)
    df = pd.DataFrame(data, columns=["x1", "x2"])
    df["label"] = _decision_boundary(data[:, 0], data[:, 1]).astype(int)
    return df


# test_make_data()


def _pseudo_residual(ys: np.ndarray, probs: np.ndarray) -> float:
    """Calculates the pseudo residual, i.e. the gradient of L wrt. f(x) at
    the mth iteration.

    Args:
        ys: an one-hot vector that encodes the ground truth labels
        probs: predicted probabilities of each class
    """
    k = np.where(ys == 1)[0][0]  # index where y_i = 1
    return probs[k]


# def _init_probs(num_instances: int, num_categories: int) -> np.ndarray:
#     """Initializes prediction probabilities for each instance and each
#     category."""
#     prob = 1 / num_categories
#     return np.ones(num_instances, num_categories) * prob


# def test__init_probs() -> None:
#     actual = _init_probs(8, 4)
#     expected = np.repeat(0.25, 8)
#     np.testing.assert_allclose(actual, expected)


# test__init_probs()


def _deviance_one_instance(k: int, fs: np.ndarray) -> float:
    """Calculates multinomial deviance for a single instance.

    Args:
        k: the index of the category to which the instance belongs to.
        fs: the prediction of all categories for this instance.
    """
    return fs[k] + np.log(np.sum(np.exp(fs)))


# def _calc_log_odds(y: np.ndarray) -> np.ndarray:
#     vals, counts = np.unique(y, return_counts=True)
#     out = np.zeros(shape=(len(y), len(vals))).astype(float)
#     for k, (val, count) in enumerate(zip(vals, counts)):
#         out[y == val, k] = (count / len(y)) / (len(y) - count / len(y))
#     return out


# def test__calc_log_odds() -> None:
#     actual = _calc_log_odds(np.array([0, 1, 1, 2]))
#     expected = np.array(
#         [
#             [1 / 3, 1, 1 / 3],
#             [1 / 3, 2 / 4, 0],
#             [0, 2 / 4, 0],
#             [0, 0, 1 / 4],
#         ]
#     )
#     np.testing.assert_allclose(actual, expected)


# test__calc_log_odds()


def _softmax(vals: np.ndarray) -> np.ndarray:
    """Convert vals to probabilities via softmax.

    Args:
        vals: shape: num_instances x num_categories
    """
    exp = np.exp(vals)
    return exp / exp.sum(axis=1, keepdims=True)


def test__softmax() -> None:
    actual = _softmax(np.array([[0, 0, 0], [0, 1, 2]]))
    expected = np.array(
        [
            [0.33333333, 0.33333333, 0.33333333],
            [0.09003057, 0.24472847, 0.66524096],
        ]
    )
    np.testing.assert_allclose(actual, expected)


test__softmax()


class GradientBoostingClassification:
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
        self.estimators: List[sklearn.tree.DecisionTreeClassifier] = []
        self.loss_history: List[float] = []
        self.loss_val_history: List[float] = []

    def fit(self, X, y) -> None:
        """Implements Algorithm 10.4.

        D: number of features per instance
        N: number of instances
        K: number of categories

        Args:
            X: instances: N x D
            y: one-hot encoded labels: N x K
        """
        self._reset()

        # init predictions f_k0(x)
        preds = np.zeros(shape=(len(X), y.shape[1]))

        for m in range(self.n_estimators):
            # convert f(x) to p(x)
            probs = _softmax(preds)

            residuals = y - probs

            estimator = sklearn.tree.DecisionTreeRegressor(max_depth=self.max_depth)
            # Note, DecisionTreeRegressor can fit multiple ys in one call to
            # .fit:
            # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor.fit
            estimator.fit(X, residuals)

            # TODO: figure out if the DecisionTreeRegressor.predict
            # indeed calculates the \gamma_{jkm} in Algorithm 10.4.
            preds += self._learning_rate * estimator.predict(X)

            self.estimators.append(estimator)

        # # predict log_odds
        # residuals = _calc_log_odds(y)

        # for cat in num_categories:

        #     for k in iterator:
        #         estimator = sklearn.tree.DecisionTreeRegressor(max_depth=self.max_depth)
        #         estimator.fit(X, residuals)
        #         self.estimators.append(estimator)

        #         preds += self._learning_rate * estimator.predict(X)

        #         residuals = y - preds
        #         loss = (residuals ** 2).mean()
        #         self.loss_history.append(loss)

    def predict_proba(self, X, verbose=True) -> np.ndarray:
        preds = None
        estimators = tqdm(self.estimators) if verbose else self.estimators
        for estimator in estimators:
            if preds is None:
                preds = estimator.predict(X)
            else:
                preds += self._learning_rate * estimator.predict(X)
        return _softmax(preds)

    def predict(self, X) -> None:
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)
