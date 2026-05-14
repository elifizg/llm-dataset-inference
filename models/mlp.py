# models/mlp.py
# Improved MLP-based membership regressor for dataset inference.
# Replaces the linear regressor (models/regressor.py) with a neural network
# that uses BCE loss, batch normalization, learning rate scheduling,
# and early stopping for more reliable training.

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from models.regressor import remove_outliers
import parameters as P


class ImprovedMLP(nn.Module):
    """
    Two-layer MLP for binary membership score prediction.

    Architecture:
        Linear(input_dim, hidden_dim) -> BatchNorm -> ReLU -> Dropout(0.3)
        Linear(hidden_dim, hidden_dim//2) -> BatchNorm -> ReLU -> Dropout(0.2)
        Linear(hidden_dim//2, 1) -> Sigmoid

    The Sigmoid output maps scores to [0, 1]:
        Score near 0 -> likely member (seen during training)
        Score near 1 -> likely non-member

    Compared to the baseline MLP in the original paper:
        - BatchNorm1d stabilizes training across seeds
        - Sigmoid + BCE loss is more appropriate than MSE for binary targets
        - Larger hidden_dim (64 vs 32) captures more feature interactions

    Args:
        input_dim:  Number of input MIA features (default: 16).
        hidden_dim: Size of the first hidden layer (default: 64).
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64):
        """Initialize the MLP with the given input and hidden dimensions."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Membership scores of shape (batch_size,), values in [0, 1].
        """
        return self.net(x).squeeze(-1)


class ImprovedMLPRegressor:
    """
    Training and inference wrapper for ImprovedMLP.

    Provides the same interface as MembershipRegressor (models/regressor.py)
    so it can be used as a drop-in replacement in the dataset inference pipeline.

    Key improvements over the baseline linear regressor:
        - Binary cross-entropy (BCE) loss instead of MSE.
        - ReduceLROnPlateau scheduler: halves the learning rate when loss
          plateaus, allowing fine-grained convergence.
        - Early stopping: restores the best model weights when validation
          loss stops improving, preventing overfitting.
        - Outlier removal and StandardScaler normalization (same as linear).

    Attributes:
        input_dim:  Number of input MIA features.
        hidden_dim: Size of the first hidden layer.
        lr:         Initial learning rate for Adam optimizer.
        patience:   Number of epochs without improvement before stopping.
        max_epochs: Maximum number of training epochs.
        scaler:     StandardScaler fitted on A-split features.
        model:      The trained ImprovedMLP instance (None before fit).
        is_fitted:  True after fit() has been called successfully.
    """

    def __init__(self,
                 input_dim: int = 16,
                 hidden_dim: int = 64,
                 lr: float = 1e-3,
                 patience: int = 20,
                 max_epochs: int = 500):
        """
        Initialize the regressor with training hyperparameters.

        Args:
            input_dim:  Number of input MIA features.
            hidden_dim: Size of the first hidden layer.
            lr:         Initial learning rate for the Adam optimizer.
            patience:   Early stopping patience in epochs.
            max_epochs: Maximum number of training epochs.
        """
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.lr         = lr
        self.patience   = patience
        self.max_epochs = max_epochs
        self.scaler     = StandardScaler()
        self.model      = None
        self.is_fitted  = False

    def fit(self,
            member_features: np.ndarray,
            nonmember_features: np.ndarray,
            verbose: bool = True) -> list:
        """
        Train the MLP on A-split features.

        Steps:
            1. Remove outliers (top/bottom 2.5% per feature).
            2. Normalize features with StandardScaler.
            3. Train with BCE loss and Adam optimizer.
            4. Halve learning rate when loss plateaus (ReduceLROnPlateau).
            5. Stop early if loss does not improve for `patience` epochs.
            6. Restore the weights from the epoch with the best loss.

        Args:
            member_features:    Feature matrix for member texts, shape (n, input_dim).
                                Assigned label 0.
            nonmember_features: Feature matrix for non-member texts, shape (n, input_dim).
                                Assigned label 1.
            verbose:            If True, print loss every 100 epochs and on early stop.

        Returns:
            loss_history: List of BCE loss values, one per training epoch.
        """
        mem_clean    = remove_outliers(member_features)
        nonmem_clean = remove_outliers(nonmember_features)

        X = np.vstack([mem_clean, nonmem_clean])
        y = np.array([0.0] * len(mem_clean) + [1.0] * len(nonmem_clean))

        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y)

        self.model = ImprovedMLP(self.input_dim, self.hidden_dim)
        optimizer  = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        criterion    = nn.BCELoss()
        best_loss    = float("inf")
        patience_cnt = 0
        best_state   = None
        loss_history = []

        self.model.train()
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            pred = self.model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            loss_val = loss.item()
            loss_history.append(loss_val)

            if loss_val < best_loss - 1e-5:
                best_loss    = loss_val
                patience_cnt = 0
                best_state   = {k: v.clone()
                                for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1

            if verbose and epoch % 100 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch:4d} | Loss: {loss_val:.4f} "
                      f"| LR: {lr_now:.6f} | Patience: {patience_cnt}")

            if patience_cnt >= self.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best loss: {best_loss:.4f})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.is_fitted = True
        print(f"[mlp] Training complete. Best loss: {best_loss:.4f}")
        return loss_history

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict membership scores for B-split features.

        Outliers are removed and features are normalized using the scaler
        fitted during training before inference.

        Args:
            features: Feature matrix of shape (n_samples, input_dim).

        Returns:
            Membership scores as a 1D numpy array of shape (n_samples,).
            Values near 0 indicate members; values near 1 indicate non-members.

        Raises:
            AssertionError: If fit() has not been called yet.
        """
        assert self.is_fitted, "Call fit() before predict()."
        features_clean = remove_outliers(features)
        X_scaled       = self.scaler.transform(features_clean)
        X_tensor       = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            scores = self.model(X_tensor).numpy()
        return scores