import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class FittedLinearProbe:
    def __init__(self, pipeline: Pipeline, n: int):
        self.pipeline = pipeline
        self.n = n

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (M, n) log-probabilities.
        Flatten X if 3D.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.pipeline.predict_log_proba(X)


class LinearProbeFactory:
    """Reference DecoderFactory. Flattens (C, T) -> (C*T,) and fits multinomial LR."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 0):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray, n: int) -> FittedLinearProbe:
        """
        X shape (M, C, T) or (M, D). y shape (M,), values in [0, n).
        Flatten X if 3D. Fit StandardScaler + LogisticRegression(multi_class='multinomial').
        Raise ValueError if any class in [0, n) has zero samples.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        for c in range(n):
            if not np.any(y == c):
                raise ValueError(f"Class {c} has zero samples")

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver="lbfgs",
            )),
        ])
        pipeline.fit(X, y)
        return FittedLinearProbe(pipeline, n)
