from typing import List, Optional, Sequence, Tuple

import numpy as np
import sklearn.tree
from tqdm import tqdm

np.random.seed(0)


def make_labels(X: np.ndarray, cutoff=9.34) -> np.ndarray:
    """Makes labels of 1 or -1 for each row in X.

    Args:
        X: a 2D array of independent standard Gaussian variables.
        cutoff: if the sum of elements squared in each row is larger than this cutoff,
            the label is 1; otherwise, -1. By default, the cutoff is \chi^2_{10}(0.5) = 9.34.
    """
    y = np.ones(shape=len(X))
    y[(X ** 2).sum(axis=1) <= cutoff] = -1
    return y


def make_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make training and testing data for the experiment."""
    X_train = np.random.normal(loc=0, scale=1, size=(2000, 10))
    X_test = np.random.normal(loc=0, scale=1, size=(10_000, 10))

    y_train = make_labels(X_train)
    y_test = make_labels(X_test)

    return X_train, y_train, X_test, y_test


def train_a_base_learner(
    instances: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
) -> sklearn.tree.DecisionTreeClassifier:
    """Train a base learner, i.e. a tree stump."""
    stump = sklearn.tree.DecisionTreeClassifier(max_depth=1)
    stump.fit(instances, labels, sample_weight=sample_weights)
    return stump


class AdaBoostM1Classifier:
    def __init__(
        self,
        learners: Sequence[sklearn.tree.DecisionTreeClassifier],
        alphas: np.ndarray,
    ) -> None:
        assert len(alphas) == len(
            learners
        ), f"number of learners ({len(learners)}) must be equal to that of alphas ({len(alphas)})"

        self._alphas = alphas
        self._learners = learners

    @property
    def learners(self) -> Sequence[sklearn.tree.DecisionTreeClassifier]:
        """Returns the list of base learners."""
        return self._learners

    @property
    def alphas(self) -> np.ndarray:
        """Returns the coefficient for each base learner."""
        return self._alphas

    def predict(
        self,
        instances: np.ndarray,
        use_n_learners: Optional[int] = None,
    ) -> np.ndarray:
        """Makes predictions on instances.

        Args:
            instances: instances to make predictions on.
            use_n_learners: only use n learners instead of all learners collected.
        """
        if use_n_learners is None:
            use_n_learners = len(self.learners)
        else:
            assert use_n_learners <= len(
                self.learners
            ), f"you can only use at most {len(self.learners)} learners"

        scaled_predictions_per_learner = []
        for k, (alpha, learner) in enumerate(zip(self.alphas, self.learners)):
            if k == use_n_learners:
                break

            scaled_predictions_per_learner.append(alpha * learner.predict(instances))

        return np.sign(np.sum(scaled_predictions_per_learner, axis=0))


def adaboost_m1(
    instances: np.ndarray,
    labels: np.ndarray,
    num_learners: int,
) -> List[Tuple[float, sklearn.tree.DecisionTreeClassifier]]:
    """Implements the AdaBoost.M1 algorithm.

    Args:
        instances: a 2D array of training instances.
        labels: an 1D array of -1 and 1, i.e. training labels.
        num_learners: number of base learners to train.

    Returns:
        An instance of AdaBoostM1Classifier.

    """
    num_instances = len(instances)

    # initial sample weights
    weights = np.repeat(1 / len(instances), num_instances)

    # collection of learners and their coefficients
    learners, alphas = [], []

    for m in tqdm(range(num_learners)):
        learner = train_a_base_learner(instances, labels, weights)

        predictions = learner.predict(instances)

        err = ((labels != predictions) * weights).sum() / weights.sum()

        alpha = np.log(1 - err) - np.log(err)

        alphas.append(alpha)
        learners.append(learner)

        weights *= np.exp(alpha * (labels != predictions))

    return AdaBoostM1Classifier(learners, alphas)
