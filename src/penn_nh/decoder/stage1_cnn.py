import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class Stage1Config:
    arch: Literal["eegnet", "dense"] = "eegnet"
    # EEGNet params
    f1: int = 8
    d: int = 2
    kernel_time: int = 16
    dropout: float = 0.25
    # Dense params
    dense_channels: int = 128
    dense_kernel: int = 3
    dense_layers: int = 2
    # Training
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: Optional[str] = None  # auto-select if None
    seed: int = 0


def _auto_device(override: Optional[str]) -> torch.device:
    if override is not None:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EEGNet(nn.Module):
    """
    Depthwise-separable temporal + spatial. Input (B, 1, C, T).
    Block 1: Conv2d(1, F1, (1, kernel_time), padding same) -> BN ->
             DepthwiseConv2d(F1, F1*D, (C, 1)) -> BN -> ELU ->
             AvgPool((1,4)) -> Dropout
    Block 2: SeparableConv2d(F1*D, F1*D, (1, kernel_time//4)) -> BN ->
             ELU -> AvgPool((1,2)) -> Dropout
    Flatten -> Linear(-, n_classes). Embedding = flatten output pre-linear.
    """

    def __init__(
        self, n_channels: int, n_times: int, n_classes: int, cfg: Stage1Config
    ):
        super().__init__()
        F1 = cfg.f1
        D = cfg.d
        F2 = F1 * D
        kt = cfg.kernel_time
        kt2 = max(kt // 4, 1)

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kt), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(cfg.dropout)

        # Block 2 — separable
        self.sep_depth = nn.Conv2d(F2, F2, (1, kt2), groups=F2, bias=False)
        self.sep_point = nn.Conv2d(F2, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 2))
        self.drop2 = nn.Dropout(cfg.dropout)

        # Explicit same-padding tuples (ONNX-safe, supports even kernels)
        self._pad1 = ((kt - 1) // 2, kt // 2)
        self._pad2 = ((kt2 - 1) // 2, kt2 // 2)

        # Embedding dimension: convs preserve T, only pools reduce it
        t_after = n_times // 4 // 2
        self.d_emb = F2 * t_after
        self.head = nn.Linear(self.d_emb, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x shape (B, C, T). Returns (logits (B, n_classes), embedding (B, d_emb))."""
        x = x.unsqueeze(1)  # (B, 1, C, T)

        # Block 1
        x = F.pad(x, self._pad1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.pad(x, self._pad2)
        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        emb = x.flatten(1)
        logits = self.head(emb)
        return logits, emb


class DenseCNN(nn.Module):
    """
    dense_layers of Conv1d(dense_channels, kernel=dense_kernel, padding=same)
    + BN + ReLU.  First layer in_channels=C.
    Flatten (not GAP) -> Linear -> logits.
    Embedding = flatten pre-linear.
    """

    def __init__(
        self, n_channels: int, n_times: int, n_classes: int, cfg: Stage1Config
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_channels
        k = cfg.dense_kernel
        pad = (k - 1) // 2
        for _ in range(cfg.dense_layers):
            layers.extend(
                [
                    nn.Conv1d(in_ch, cfg.dense_channels, k, padding=pad),
                    nn.BatchNorm1d(cfg.dense_channels),
                    nn.ReLU(),
                ]
            )
            in_ch = cfg.dense_channels
        self.features = nn.Sequential(*layers)
        self.d_emb = cfg.dense_channels * n_times
        self.head = nn.Linear(self.d_emb, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        emb = x.flatten(1)
        logits = self.head(emb)
        return logits, emb


class Stage1Factory:
    """DecoderFactory-compatible wrapper."""

    def __init__(self, cfg: Stage1Config = None):
        self.cfg = cfg or Stage1Config()

    def fit(self, X: np.ndarray, y: np.ndarray, n: int) -> "FittedStage1":
        """
        X (M, C, T) float32. y (M,) int in [0, n).
        Build model per cfg.arch. Train with AdamW, cross-entropy, linear LR
        decay to 0. Seed torch, numpy, cuda for reproducibility.
        """
        cfg = self.cfg
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 3:
            raise ValueError("X must be 3D (M, C, T)")
        if X.shape[0] == 0:
            raise ValueError("M must be > 0")
        classes_present = set(int(c) for c in np.unique(y))
        missing = sorted(set(range(n)) - classes_present)
        if missing:
            raise ValueError(f"Missing classes: {missing}")

        # Reproducibility
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        device = _auto_device(cfg.device)
        M, C, T = X.shape

        if cfg.arch == "eegnet":
            model = EEGNet(C, T, n, cfg).to(device)
        elif cfg.arch == "dense":
            model = DenseCNN(C, T, n, cfg).to(device)
        else:
            raise ValueError(f"Unknown arch: {cfg.arch}")

        X_t = torch.from_numpy(X).to(device)
        y_t = torch.from_numpy(y).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda ep: max(1.0 - ep / max(cfg.epochs, 1), 0.0),
        )

        model.train()
        for _ in range(cfg.epochs):
            perm = torch.randperm(M, device=device)
            for start in range(0, M, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                logits, _ = model(X_t[idx])
                loss = F.cross_entropy(logits, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        return FittedStage1(model, cfg, device, n)


class FittedStage1:
    def __init__(
        self, model: nn.Module, cfg: Stage1Config, device: torch.device, n: int
    ):
        self.model = model.eval()
        self.cfg = cfg
        self.device = device
        self.n = n

    def _forward_torch(self, X: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            return self.model(X_t)

    def score(self, X: np.ndarray) -> np.ndarray:
        """(M, C, T) -> (M, n) log-softmax, numpy float32."""
        logits, _ = self._forward_torch(X)
        return F.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (logits (M, n), embeddings (M, d_emb))."""
        logits, emb = self._forward_torch(X)
        return logits.cpu().numpy(), emb.cpu().numpy()

    def export_onnx(self, path: str, example_shape: tuple[int, int, int]) -> None:
        """
        Export with dynamic batch axis. example_shape = (1, C, T).
        Output names: ["logits", "embedding"]. Input name: "input".
        opset_version=17.
        """
        dummy = torch.randn(*example_shape).to(self.device)
        torch.onnx.export(
            self.model,
            dummy,
            path,
            input_names=["input"],
            output_names=["logits", "embedding"],
            dynamic_axes={
                "input": {0: "batch"},
                "logits": {0: "batch"},
                "embedding": {0: "batch"},
            },
            opset_version=17,
        )
