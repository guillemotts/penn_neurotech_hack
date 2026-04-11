import numpy as np


def fit_prototypes(embeddings: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    """
    embeddings: (M, d). labels: (M,) int in [0, n).
    Returns (n, d) L2-normalized class-mean prototypes.
    Empty classes: raise ValueError naming the missing class ids.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels, dtype=int)
    d = embeddings.shape[1]
    protos = np.zeros((n, d), dtype=float)

    missing = []
    for c in range(n):
        mask = labels == c
        if not np.any(mask):
            missing.append(c)
            continue
        protos[c] = embeddings[mask].mean(axis=0)

    if missing:
        raise ValueError(f"Missing classes: {missing}")

    norms = np.linalg.norm(protos, axis=1, keepdims=True)
    protos = protos / norms
    return protos


def cosine_scores(e: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """
    e: (d,) or (B, d). protos: (N, d), assumed L2-normalized.
    L2-normalize e along last axis. Return (N,) or (B, N) cosine similarities.
    """
    e = np.asarray(e, dtype=float)
    single = e.ndim == 1
    if single:
        e = e[np.newaxis, :]

    norms = np.linalg.norm(e, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    e_normed = e / norms

    sim = e_normed @ protos.T

    if single:
        return sim[0]
    return sim


def _stable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    x_max = x.max(axis=-1, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / exps.sum(axis=-1, keepdims=True)


def combine(logits: np.ndarray, cos: np.ndarray, alpha: float, tau: float) -> np.ndarray:
    """
    Returns alpha * softmax(logits) + (1 - alpha) * softmax(cos / tau).
    Shapes must match. 0 <= alpha <= 1, tau > 0. ValueError otherwise.
    """
    logits = np.asarray(logits, dtype=float)
    cos = np.asarray(cos, dtype=float)

    if logits.shape != cos.shape:
        raise ValueError("logits and cos must have the same shape")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be in [0, 1]")
    if tau <= 0:
        raise ValueError("tau must be > 0")

    return alpha * _stable_softmax(logits) + (1 - alpha) * _stable_softmax(cos / tau)
