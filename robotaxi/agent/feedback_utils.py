from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import sys
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.scale import FuncScale
import numpy as np

# Allow importing project-root `utils.py` when executed as a script
try:
    from utils import parse_grid
except Exception:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from utils import parse_grid
from robotaxi.agent.tamer_agent import compute_delta_features_v4, compute_delta_features_v6
from robotaxi.agent.tamer_agent import compute_delta_features_v4, compute_delta_features_v6

# ===== Global visualization and kernel flags =====
# Toggle linear vs. symmetric-log color scaling for the heatmap (adjusted feedback).
# For your request, we keep adjusted feedback linear by default.
USE_LINEAR_COLOR: bool = True

USE_V6: bool = True
# Axis scale mode for X1/X2 axes. Options:
# - 'linear': no scaling (default previously)
# - 'symlog': symmetric log with a linear region near 0
# - 'reciprocal': monotonic reciprocal-like squashing: f(x)=sign(x)*(1-1/(|x|+1))
#                 inverts with f^{-1}(y)=sign(y)*(1/(1-|y|)-1). This compresses large |x| toward ±1.
AXIS_SCALE_MODE: str = 'linear'
LOG_AXES_LINTHRESH: float = 1e-2  # linear region around 0 for symlog axes

# SymLog linthresh for color if/when linear color is disabled
SYMLIN_COLOR_LINTHRESH: float = 1e-2

# Zoom window half-width for the plotted view; we show [-PLOT_ZOOM_LIMIT, +PLOT_ZOOM_LIMIT]^2
PLOT_ZOOM_LIMIT: float = 3

# Global multiplier to scale the effective RBF kernel length scale ("radius-related" constant)
RBF_LENGTH_SCALE_MULT: float = 1.0

# For reciprocal axis mode: denominators to be evenly spaced on the axis (positive side)
# Example: [1, 2, 3, 8] will place x = 1, 1/2, 1/3, 1/8 at equal visual spacing.
RECIP_KNOTS_DENOMS: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]

class FeedbackPreProcessor(ABC):
    """
    This class keeps statistics of feedbacks, and supports an add function that add feedbacks to the statistics, and get_feedback that returns the adjusted feedback
    usually the two functions are called next to each other
    """    
    @abstractmethod
    def add_feedback(self, state, action, next_state, noisy_reward) -> None:
        pass
    
    @abstractmethod
    def get_feedback(self, state, action, next_state, noisy_reward) -> float:
        # return the adjusted reward
        pass
    
class visualizeFeedbackV4(ABC):
    @abstractmethod
    def visualize_current_processor(self, save_path: str="", show_plot: bool=False) -> None:
        # renders a grid where X1 and X2 correspond to two v4 feature dimensions and Y is the adjusted feedback
        # Plot a 2D heatmap over the domain [-2, 2] x [-2, 2] with values in [-1, 1]
        # Also overlay observed datapoints (+: positive, o: zero, _: negative)
        pass
    
    
    



# ===== Utilities =====

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float) -> np.ndarray:
    """RBF (SE) kernel."""
    X1 = np.atleast_2d(X1); X2 = np.atleast_2d(X2)
    d2 = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=-1)
    eff_ls = float(length_scale) * float(RBF_LENGTH_SCALE_MULT)
    return np.exp(-0.5 * d2 / (eff_ls**2 + 1e-12))

def _apply_axis_scale(ax):
    """Apply the selected axis scale mode to both axes.

    Modes:
      - linear: no change
      - symlog: symmetric log with linear threshold LOG_AXES_LINTHRESH
      - reciprocal: monotonic reciprocal-like scale defined by functions below
    """
    mode = (AXIS_SCALE_MODE or 'linear').lower()
    if mode == 'linear':
        return
    if mode == 'symlog':
        ax.set_xscale('symlog', linthresh=LOG_AXES_LINTHRESH)
        ax.set_yscale('symlog', linthresh=LOG_AXES_LINTHRESH)
        return
    if mode == 'reciprocal':
        # Build piecewise-linear mapping on |x| with knots at reciprocals of denominators
        denoms = [d for d in RECIP_KNOTS_DENOMS if d > 0]
        if len(denoms) == 0:
            return
        x_knots = sorted([1.0/float(d) for d in denoms])  # ascending in (0,1]
        y_knots = np.linspace(0.0, 1.0, num=len(x_knots))  # equally spaced positions

        def interp_piecewise(a, xp, fp):
            # clamp outside range
            return np.interp(a, xp, fp, left=fp[0], right=fp[-1])

        def fwd(v):
            vv = np.asarray(v, dtype=float)
            s = np.sign(vv)
            a = np.abs(vv)
            y = interp_piecewise(a, np.array(x_knots), np.array(y_knots))
            return s * y

        def inv(w):
            ww = np.asarray(w, dtype=float)
            s = np.sign(ww)
            a = np.abs(ww)
            x = interp_piecewise(a, np.array(y_knots), np.array(x_knots))
            return s * x

        try:
            ax.set_xscale('function', functions=(fwd, inv))
            ax.set_yscale('function', functions=(fwd, inv))
        except Exception:
            ax.set_xscale(FuncScale(ax, functions=(fwd, inv)))
            ax.set_yscale(FuncScale(ax, functions=(fwd, inv)))

        # Place symmetric ticks at the specified reciprocals
        pos_ticks = sorted([1.0/float(d) for d in denoms])
        neg_ticks = [-t for t in reversed(pos_ticks)]
        xticks = neg_ticks + pos_ticks
        yticks = xticks
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        # Labels like 1, 1/2, 1/3, 1/8 (and negatives)
        def frac_label(t):
            if abs(t) < 1e-12:
                return '0'
            sgn = '-' if t < 0 else ''
            val = abs(t)
            for d in denoms:
                if abs(val - 1.0/float(d)) < 1e-9:
                    return f"{sgn}1/{d}" if d != 1 else f"{sgn}1"
            return f"{sgn}{val:.3f}"
        ax.set_xticklabels([frac_label(t) for t in xticks])
        ax.set_yticklabels([frac_label(t) for t in yticks])
        return
    # Fallback: treat as linear if unknown
    return

def median_heuristic_lengthscale(X: np.ndarray, cap: int = 200) -> float:
    """Median distance heuristic / sqrt(2)."""
    X = np.asarray(X)
    if len(X) < 2:
        return 0.4
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(cap, len(X)), replace=False)
    Xs = X[idx]
    d2 = np.sum((Xs[:, None, :] - Xs[None, :, :])**2, axis=-1)
    tri = d2[np.triu_indices_from(d2, 1)]
    med = np.median(np.sqrt(np.maximum(tri, 0.0)))
    return max(float(med) / np.sqrt(2.0), 1e-3)

def clip_output(z: np.ndarray, negative_only: bool) -> np.ndarray:
    lo, hi = (-1.0, 0.0) if negative_only else (-1.0, 1.0)
    return np.clip(z, lo, hi)



# ===== Abstract interface =====

class FeedbackPreProcessor(ABC):
    """
    Keeps statistics of feedback and offers:
      - add_feedback: update internal stats/model with (s, a, s', noisy_reward)
      - get_feedback: return adjusted (denoised/smoothed) reward
    Usually called back-to-back.
    """
    @abstractmethod
    def add_feedback(self, state, action, next_state, noisy_reward) -> None:
        pass

    @abstractmethod
    def get_feedback(self, state, action, next_state, noisy_reward) -> float:
        pass
    
class visualizeableFeedbackPreProcessorV4(visualizeFeedbackV4, FeedbackPreProcessor):
    pass

# ===== Gaussian-Process-based preprocessor =====
class GPPFeedbackPreProcessor(visualizeableFeedbackPreProcessorV4):
    """
    GP regression with RBF kernel (mean prediction only).
    Stores all (feature, noisy_label) pairs; on prediction, computes k(x, X) K^-1 y.
    """
    def __init__(
        self,
        feature_version: str = "v4",
        negative_feedback_only: bool = False,
        length_scale: Optional[float] = None,
        noise_sigma: float = 0.20,       # observation noise (sigma)
        ridge_jitter: float = 1e-9       # numerical stability
    ):
        assert feature_version in ["v4"], "Invalid feature version"
        self.feature_version = feature_version
        self.negative_feedback_only = negative_feedback_only
        self.featurize_fn = lambda state, next_state: compute_delta_features_v4(state, next_state)[2:4]
        if USE_V6:
            self.featurize_fn = lambda state, next_state: compute_delta_features_v6(state, next_state)[2:4]

        # Hyperparams
        self.user_length_scale = length_scale
        self.noise_sigma = float(noise_sigma)
        self.ridge_jitter = float(ridge_jitter)

        # Replay buffer
        self._X: List[np.ndarray] = []
        self._y: List[float] = []

        # Cached factors after fitting
        self._ls_: Optional[float] = None
        self._K_inv_y: Optional[np.ndarray] = None
        self._X_mat: Optional[np.ndarray] = None

    # ---- Internal helpers ----

    def _labels_ok(self, y: float) -> float:
        """Constrain incoming label to accepted set."""
        if self.negative_feedback_only:
            # Allowed: {0, -1}
            if y not in (0, -1):
                # map to nearest of {0, -1}
                y = 0 if y > -0.5 else -1
        else:
            # Allowed: {-1, 0, 1}
            if y not in (-1, 0, 1):
                # map to nearest of {-1, 0, 1}
                y = int(np.clip(np.round(y), -1, 1))
        return float(y)

    def _refit_cache(self) -> None:
        """Compute K^-1 y and cache; called lazily."""
        if len(self._X) == 0:
            self._K_inv_y = None
            self._X_mat = None
            self._ls_ = self.user_length_scale if self.user_length_scale is not None else 0.4
            return
        X = np.vstack(self._X)
        y = np.asarray(self._y, dtype=float)

        ls = self.user_length_scale if self.user_length_scale is not None else median_heuristic_lengthscale(X)
        K = rbf_kernel(X, X, ls)
        Ky = K + (self.noise_sigma**2) * np.eye(len(X)) + self.ridge_jitter * np.eye(len(X))

        # Solve Ky * alpha = y  (alpha = K^-1 y)
        alpha = np.linalg.solve(Ky, y)
        self._K_inv_y = alpha
        self._X_mat = X
        self._ls_ = float(ls)

    def _predict_mean(self, xf: np.ndarray) -> float:
        if self._X_mat is None or self._K_inv_y is None or len(self._X) == 0:
            # With no data, defer to raw noisy reward semantics: neutral
            return 0.0 if self.negative_feedback_only else 0.0
        kx = rbf_kernel(xf[None, :], self._X_mat, self._ls_).reshape(-1)  # shape [N]
        mu = float(kx @ self._K_inv_y)
        return mu

    # ---- Public API ----

    def add_feedback(self, state, action, next_state, noisy_reward) -> None:
        xf = np.asarray(self.featurize_fn(state, next_state), dtype=float).reshape(1, -1)
        y = self._labels_ok(float(noisy_reward))
        self._X.append(xf.reshape(-1))
        self._y.append(y)
        # Lazily refit (cheap here; N<=few hundreds)
        self._refit_cache()

    def get_feedback(self, state, action, next_state, noisy_reward) -> float:
        xf = np.asarray(self.featurize_fn(state, next_state), dtype=float).reshape(-1)
        mu = self._predict_mean(xf)
        return float(clip_output(mu, self.negative_feedback_only))
    
    # ---- visualization ----
    def visualize_current_processor(self, save_path: str = "", show_plot: bool = False) -> None:
        if self.feature_version != "v4":
            raise ValueError("visualize_current_processor only supports feature_version='v4'")

        # Grid over [-2, 2]^2
        g = 60
        gx = np.linspace(-2.0, 2.0, g)
        gy = np.linspace(-2.0, 2.0, g)
        GX, GY = np.meshgrid(gx, gy)
        grid2 = np.stack([GX.ravel(), GY.ravel()], axis=-1)  # shape [g*g, 2]

        # Predictions
        if len(self._X) == 0:
            Z = np.zeros((g, g))
        else:
            ls = self._ls_ if self._ls_ is not None else 0.4
            # Embed 2D grid into v4 feature space (6D): use dims 2 and 3, others zero
            d_in = self._X_mat.shape[1]
            grid_full = np.zeros((grid2.shape[0], d_in), dtype=float)
            # Map X1->feat[2], X2->feat[3]
            if d_in >= 4:
                grid_full[:, 2] = grid2[:, 0]
                grid_full[:, 3] = grid2[:, 1]
            else:
                # Fallback: if model was trained with 2D features, place directly
                grid_full = grid2[:, :d_in]
            Ks = rbf_kernel(grid_full, self._X_mat, ls)
            Z = (Ks @ self._K_inv_y).reshape(g, g)
            Z = clip_output(Z, self.negative_feedback_only)

        # Plot
        fig = plt.figure(figsize=(6, 5), dpi=120)
        ax = plt.gca()
        vmin, vmax = -1.0, (0.0 if self.negative_feedback_only else 1.0)
        if USE_LINEAR_COLOR:
            im = ax.imshow(Z, extent=(-2, 2, -2, 2), origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
        else:
            norm = SymLogNorm(linthresh=SYMLIN_COLOR_LINTHRESH, vmin=vmin, vmax=vmax)
            im = ax.imshow(Z, extent=(-2, 2, -2, 2), origin="lower", norm=norm, interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax); cbar.set_label("adjusted feedback")

        # Overlay observed points by label using requested symbols
        if len(self._X) > 0:
            Xobs = np.vstack(self._X)
            yobs = np.asarray(self._y, dtype=float).astype(int)
            for val, mk in [(1, '+'), (0, 'o'), (-1, '_')]:
                sel = (yobs == val)
                if np.any(sel):
                    # Plot using feature dims (2,3) which define the visualization axes
                    xi = 2 if Xobs.shape[1] > 2 else 0
                    yi = 3 if Xobs.shape[1] > 3 else (1 if Xobs.shape[1] > 1 else 0)
                    ax.scatter(Xobs[sel, xi], Xobs[sel, yi], s=28, marker=mk, alpha=0.9, linewidths=1.2)

        # Axes scaling and zoom
        _apply_axis_scale(ax)
        ax.set_xlim(-PLOT_ZOOM_LIMIT, PLOT_ZOOM_LIMIT); ax.set_ylim(-PLOT_ZOOM_LIMIT, PLOT_ZOOM_LIMIT)
        ax.set_xlabel("X1 (v4 Δpassenger proximity)"); ax.set_ylabel("X2 (v4 Δobstacle proximity)")
        ax.set_title("GPPFeedbackPreProcessor — current adjusted feedback")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

# ===== GP with Uncertainty (mean + variance) preprocessor =====
class GPPUQFeedbackPreProcessor(GPPFeedbackPreProcessor):
    """
    Extends GP preprocessor to visualize both mean prediction and epistemic uncertainty (diagonal variance).
    Uses the same 2D feature projection and replay buffer; computes variance via solves against Ky.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._Ky_chol = None  # cached factorization for uncertainty solves (optional)

    def _ensure_Ky(self) -> Optional[np.ndarray]:
        if self._X_mat is None or len(self._X) == 0 or self._ls_ is None:
            return None
        K = rbf_kernel(self._X_mat, self._X_mat, self._ls_)
        Ky = K + (self.noise_sigma**2) * np.eye(len(self._X_mat)) + self.ridge_jitter * np.eye(len(self._X_mat))
        return Ky

    def visualize_current_processor(self, save_path: str = "", show_plot: bool = False) -> None:
        if self.feature_version != "v4":
            raise ValueError("visualize_current_processor only supports feature_version='v4'")

        # Grid over [-2, 2]^2
        g = 60
        gx = np.linspace(-2.0, 2.0, g)
        gy = np.linspace(-2.0, 2.0, g)
        GX, GY = np.meshgrid(gx, gy)
        grid2 = np.stack([GX.ravel(), GY.ravel()], axis=-1)  # [g*g, 2]

        # Prepare inference matrices
        if len(self._X) == 0:
            Z_mean = np.zeros((g, g))
            Z_var  = np.zeros((g, g))
        else:
            ls = self._ls_ if self._ls_ is not None else 0.4
            d_in = self._X_mat.shape[1]
            grid_full = np.zeros((grid2.shape[0], d_in), dtype=float)
            if d_in >= 2:
                grid_full[:, :2] = grid2[:, :2]
            else:
                grid_full = grid2[:, :d_in]

            # Mean
            Ks = rbf_kernel(grid_full, self._X_mat, ls)  # [M,N]
            Z_mean = (Ks @ self._K_inv_y).reshape(g, g)
            Z_mean = clip_output(Z_mean, self.negative_feedback_only)

            # Variance: diag(K_ss - Ks Ky^{-1} Ks^T)
            Ky = self._ensure_Ky()
            if Ky is None:
                Z_var = np.zeros((g, g))
            else:
                # Solve Ky * V = Ks^T for V (N x M), in batches to reduce memory
                M = Ks.shape[0]
                batch = 600
                var_diag = np.empty(M, dtype=float)
                for start in range(0, M, batch):
                    end = min(M, start + batch)
                    Kst_slice = Ks[start:end, :].T  # [N, B]
                    V = np.linalg.solve(Ky, Kst_slice)  # [N, B]
                    # diag term for each column: sum(Ks_row * (V^T_row))
                    quad = np.sum(Ks[start:end, :] * V.T, axis=1)  # [B]
                    kxx = 1.0  # rbf_kernel(x,x)=1 for SE kernel
                    var_diag[start:end] = np.maximum(kxx - quad, 0.0)
                Z_var = var_diag.reshape(g, g)

        # Plot side-by-side: mean | uncertainty
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=120, constrained_layout=True)

        # Mean panel
        ax = axes[0]
        vmin, vmax = -1.0, (0.0 if self.negative_feedback_only else 1.0)
        if USE_LINEAR_COLOR:
            im0 = ax.imshow(Z_mean, extent=(-2, 2, -2, 2), origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
        else:
            norm = SymLogNorm(linthresh=SYMLIN_COLOR_LINTHRESH, vmin=vmin, vmax=vmax)
            im0 = ax.imshow(Z_mean, extent=(-2, 2, -2, 2), origin="lower", norm=norm, interpolation="nearest")
        cbar0 = plt.colorbar(im0, ax=ax); cbar0.set_label("mean adjusted feedback")
        if len(self._X) > 0:
            Xobs = np.vstack(self._X)
            yobs = np.asarray(self._y, dtype=float).astype(int)
            for val, mk in [(1, '+'), (0, 'o'), (-1, '_')]:
                sel = (yobs == val)
                if np.any(sel):
                    xi = 0
                    yi = 1 if Xobs.shape[1] > 1 else 0
                    ax.scatter(Xobs[sel, xi], Xobs[sel, yi], s=28, marker=mk, alpha=0.9, linewidths=1.2)
        _apply_axis_scale(ax)
        ax.set_xlim(-PLOT_ZOOM_LIMIT, PLOT_ZOOM_LIMIT); ax.set_ylim(-PLOT_ZOOM_LIMIT, PLOT_ZOOM_LIMIT)
        ax.set_xlabel("X1 (v4 Δpassenger proximity)"); ax.set_ylabel("X2 (v4 Δobstacle proximity)")
        ax.set_title("GP mean")

        # Uncertainty panel
        ax = axes[1]
        vmin_uq, vmax_uq = 0.0, float(np.percentile(Z_var, 95)) if np.any(Z_var) else 1.0
        im1 = ax.imshow(Z_var, extent=(-2, 2, -2, 2), origin="lower", vmin=vmin_uq, vmax=vmax_uq, interpolation="nearest", cmap="magma")
        cbar1 = plt.colorbar(im1, ax=ax); cbar1.set_label("predictive variance")
        if len(self._X) > 0:
            Xobs = np.vstack(self._X)
            yobs = np.asarray(self._y, dtype=float).astype(int)
            for val, mk in [(1, '+'), (0, 'o'), (-1, '_')]:
                sel = (yobs == val)
                if np.any(sel):
                    xi = 0
                    yi = 1 if Xobs.shape[1] > 1 else 0
                    ax.scatter(Xobs[sel, xi], Xobs[sel, yi], s=28, marker=mk, alpha=0.7, linewidths=1.0, color="white")
        _apply_axis_scale(ax)
        ax.set_xlim(-PLOT_ZOOM_LIMIT, PLOT_ZOOM_LIMIT); ax.set_ylim(-PLOT_ZOOM_LIMIT, PLOT_ZOOM_LIMIT)
        ax.set_xlabel("X1 (v4 Δpassenger proximity)"); ax.set_ylabel("X2 (v4 Δobstacle proximity)")
        ax.set_title("GP uncertainty")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

# ===== Tiny-ML (small MLP with Huber) preprocessor =====
class TINYMLFeedbackPreProcessor(visualizeableFeedbackPreProcessorV4):
    """
    Small 2-layer ReLU MLP trained with Huber loss + weight decay on the replay buffer.
    For speed and stability with small data, we refit briefly (few SGD steps) on each add/get.
    """
    def __init__(
        self,
        feature_version: str = "v4",
        negative_feedback_only: bool = False,
        hidden: int = 32,
        seed: int = 123,
        lr: float = 3e-3,
        steps_per_update: int = 100,
        batch_size: int = 64,
        delta_huber: float = 1.0,
        weight_decay: float = 2e-4
    ):
        assert feature_version in ["v4"], "Invalid feature version"
        self.feature_version = feature_version
        self.negative_feedback_only = negative_feedback_only
        self.featurize_fn = lambda state, next_state: compute_delta_features_v4(state, next_state)[2:4]
        if USE_V6:
            self.featurize_fn = lambda state, next_state: compute_delta_features_v6(state, next_state)[2:4]

        # Replay
        self._X: List[np.ndarray] = []
        self._y: List[float] = []

        # Model params
        self.hidden = int(hidden)
        self.rng = np.random.default_rng(seed)
        self.lr = float(lr)
        self.steps = int(steps_per_update)
        self.batch = int(batch_size)
        self.delta = float(delta_huber)
        self.wd = float(weight_decay)

        # Lazy init after we see input dim
        self._W1 = self._b1 = self._W2 = self._b2 = None
        self._m = self._v = None
        self._t = 0

    # ---- Internal helpers ----

    def _labels_ok(self, y: float) -> float:
        if self.negative_feedback_only:
            if y not in (0, -1):
                y = 0 if y > -0.5 else -1
        else:
            if y not in (-1, 0, 1):
                y = int(np.clip(np.round(y), -1, 1))
        return float(y)

    def _ensure_params(self, d_in: int):
        if self._W1 is not None: return
        H = self.hidden
        rs = self.rng
        # Kaiming-ish init
        self._W1 = rs.normal(0, 1/np.sqrt(d_in), size=(d_in, H))
        self._b1 = np.zeros(H)
        self._W2 = rs.normal(0, 1/np.sqrt(H), size=(H, 1))
        self._b2 = np.zeros(1)
        # Adam buffers
        self._m = [np.zeros_like(self._W1), np.zeros_like(self._b1),
                   np.zeros_like(self._W2), np.zeros_like(self._b2)]
        self._v = [np.zeros_like(x) for x in self._m]
        self._t = 0

    @staticmethod
    def _relu(x): return np.maximum(0.0, x)

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self._relu(X @ self._W1 + self._b1)          # [N,H]
        out = (h @ self._W2 + self._b2).squeeze(-1)      # [N]
        return h, out, X

    def _huber_grad(self, r: np.ndarray, delta: float) -> np.ndarray:
        absr = np.abs(r)
        return np.where(absr <= delta, r, delta*np.sign(r))

    def _adam_step(self, grads: List[np.ndarray], lr: float, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        params = [self._W1, self._b1, self._W2, self._b2]
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = b1*self._m[i] + (1-b1)*g
            self._v[i] = b2*self._v[i] + (1-b2)*(g*g)
            mhat = self._m[i]/(1-b1**self._t)
            vhat = self._v[i]/(1-b2**self._t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)

    def _train_few_steps(self):
        if len(self._X) == 0: return
        X = np.vstack(self._X)
        y = np.asarray(self._y, dtype=float)
        self._ensure_params(X.shape[1])

        N = len(X)
        bs = min(self.batch, N)
        for _ in range(self.steps):
            idx = self.rng.choice(N, size=bs, replace=False)
            Xb = X[idx]; yb = y[idx]
            h, out, _ = self._forward(Xb)
            r = out - yb
            grad_out = self._huber_grad(r, self.delta) / bs  # mean over batch

            # Backprop
            dW2 = (h.T @ grad_out[:, None]) + self.wd*self._W2
            db2 = grad_out.sum(0, keepdims=True)

            gh = grad_out[:, None] @ self._W2.T
            gh[h <= 0.0] = 0.0

            dW1 = (Xb.T @ gh) + self.wd*self._W1
            db1 = gh.sum(0)

            self._adam_step([dW1, db1, dW2, db2], self.lr)

    def _predict(self, xf: np.ndarray) -> float:
        if self._W1 is None or len(self._X) == 0:
            return 0.0 if self.negative_feedback_only else 0.0
        h, out, _ = self._forward(xf[None, :])
        return float(out[0])

    # ---- Public API ----

    def add_feedback(self, state, action, next_state, noisy_reward) -> None:
        xf = np.asarray(self.featurize_fn(state, next_state), dtype=float).reshape(1, -1)
        y  = self._labels_ok(float(noisy_reward))
        self._X.append(xf.reshape(-1))
        self._y.append(y)
        # Brief online update for freshness
        self._train_few_steps()

    def get_feedback(self, state, action, next_state, noisy_reward) -> float:
        # Optionally do a brief update to incorporate latest pair before readout
        self._train_few_steps()
        xf = np.asarray(self.featurize_fn(state, next_state), dtype=float).reshape(-1)
        z = self._predict(xf)
        return float(clip_output(z, self.negative_feedback_only))
    
    def visualize_current_processor(self, save_path: str = "", show_plot: bool = False) -> None:
        if self.feature_version != "v4":
            raise ValueError("visualize_current_processor only supports feature_version='v4'")

        # Grid over [-2, 2]^2
        g = 60
        gx = np.linspace(-2.0, 2.0, g)
        gy = np.linspace(-2.0, 2.0, g)
        GX, GY = np.meshgrid(gx, gy)
        grid2 = np.stack([GX.ravel(), GY.ravel()], axis=-1)

        # Predictions
        if len(self._X) == 0 or self._W1 is None:
            Z = np.zeros((g, g))
        else:
            # Embed 2D point into v4 feature space (6D): use dims 2 and 3, others zero
            d_in = self._W1.shape[0]
            def embed(pt2):
                full = np.zeros(d_in, dtype=float)
                if d_in >= 4:
                    full[2] = pt2[0]
                    full[3] = pt2[1]
                else:
                    full[:min(2, d_in)] = pt2[:min(2, d_in)]
                return full
            Z = np.array([self._predict(embed(pt)) for pt in grid2]).reshape(g, g)
            Z = clip_output(Z, self.negative_feedback_only)

        # Plot
        fig = plt.figure(figsize=(6, 5), dpi=120)
        ax = plt.gca()
        vmin, vmax = -1.0, (0.0 if self.negative_feedback_only else 1.0)
        if USE_LINEAR_COLOR:
            im = ax.imshow(Z, extent=(-2, 2, -2, 2), origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
        else:
            norm = SymLogNorm(linthresh=SYMLIN_COLOR_LINTHRESH, vmin=vmin, vmax=vmax)
            im = ax.imshow(Z, extent=(-2, 2, -2, 2), origin="lower", norm=norm, interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax); cbar.set_label("adjusted feedback")

        # Overlay observed points with requested markers
        if len(self._X) > 0:
            Xobs = np.vstack(self._X)
            yobs = np.asarray(self._y, dtype=float).astype(int)
            for val, mk in [(1, '+'), (0, 'o'), (-1, '_')]:
                sel = (yobs == val)
                if np.any(sel):
                    xi = 2 if Xobs.shape[1] > 2 else 0
                    yi = 3 if Xobs.shape[1] > 3 else (1 if Xobs.shape[1] > 1 else 0)
                    ax.scatter(Xobs[sel, xi], Xobs[sel, yi], s=28, marker=mk, alpha=0.9, linewidths=1.2)

        # Axes scaling and zoom
        _apply_axis_scale(ax)
        ax.set_xlim(-PLOT_ZOOM_LIMIT, PLOT_ZOOM_LIMIT); ax.set_ylim(-PLOT_ZOOM_LIMIT, PLOT_ZOOM_LIMIT)
        ax.set_xlabel("X1 (v4 Δpassenger proximity)"); ax.set_ylabel("X2 (v4 Δobstacle proximity)")
        ax.set_title("TINYMLFeedbackPreProcessor — current adjusted feedback")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show_plot:
            plt.show()
        else:
            plt.close(fig)




def check_feedback_processor(
    states: Sequence[Any],
    next_states: Sequence[Any],
    rewards: Sequence[float],
    feedback_processor: visualizeableFeedbackPreProcessorV4,
) -> None:
    """
    Incrementally feed (state, next_state, reward) to the processor and visualize at milestones.
    Milestones: 10, 20, 30, 40, 50, 100, 150, 200, and then every +100 up to N.

    Saves images into ./feedback_v4_previews/ as PNGs.
    """
    # --- Basic validation ---
    n = len(states)
    if not (len(next_states) == n and len(rewards) == n):
        raise ValueError("states, next_states, and rewards must have the same length.")

    # Prepare milestone indices
    base_milestones = {10, 20, 30, 40, 50, 100, 150, 200}
    tail_milestones = set(range(300, n + 1, 100)) if n >= 300 else set()
    milestones = sorted((base_milestones | tail_milestones) & set(range(1, n + 1)))

    # Output directory
    out_dir = Path("./feedback_v4_previews")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stream and visualize
    for i in range(n):
        s, ns, r = states[i], next_states[i], rewards[i]
        # action not used in v4 processors; pass None by convention
        feedback_processor.add_feedback(s, None, ns, r)

        step = i + 1
        if step in milestones:
            # safer file name with zero-padded step id
            save_path = str(out_dir / f"vis_step_{step:05d}.png")
            feedback_processor.visualize_current_processor(save_path=save_path, show_plot=False)
            

    # Optional: final snapshot if last step wasn't a milestone
    if n not in milestones:
        save_path = str(out_dir / f"vis_step_{n:05d}.png")
        try:
            feedback_processor.visualize_current_processor(save_path=save_path, show_plot=False)
        except Exception as e:
            print(f"[check_feedback_processor] Warning: final visualization failed at step {n}: {e}")
            
       
       
class UserSignalMixer(ABC):
    """
    A class that add noise to the user signals
    """
    @abstractmethod
    def mix_signal(self, signal: float) -> float:
        pass
       
       
class DiscreteNegativeOnlySignalMixer(UserSignalMixer):
    """
    given a signal in {0, -1}, mix it with the tpr and tnr, return a signal in {0, -1}
    """
    def __init__(self, tpr: float, tnr: float):
        self.tpr = tpr
        self.tnr = tnr
        
    def mix_signal(self, signal: float) -> float:
        assert signal in {0, -1}, "signal must be in {0, -1}"
        if signal == -1:
            return -1 if np.random.rand() <= self.tpr else 0
        else:
            return 0 if np.random.rand() <= self.tnr else -1

class DiscreteNegativeOnlySignalMixerWithContinuousOutput(UserSignalMixer):
    """
    given a signal in {0, -1}, assume some distribution of the signal, output a signal in [-1, 0]
    for example, if the signal is -1, the output is from N(mu_p, sigma_p), if the signal is 0, the output is from N(mu_n, sigma_n)
    note: it might be confusing but in our language -1 is positive feedback
    """
    def __init__(self, mu_p: float, sigma_p: float, mu_n: float, sigma_n: float):
        self.mu_p = mu_p
        self.sigma_p = sigma_p
        self.mu_n = mu_n
        self.sigma_n = sigma_n
        assert mu_p < mu_n, "mu_p must be less than mu_n, please check the notes, positive feedback is -1"
        
    def mix_signal(self, signal: float) -> float:
        """
        signal: -1 or 0
        return: [-1, 0] # continuous
        """
        # clip signal to [-1, 0] before returning
        assert signal in {0, -1}, "signal must be in {0, -1}"
        if signal == -1:
            return np.clip(np.random.normal(self.mu_p, self.sigma_p), -1, 0)
        else:
            return np.clip(np.random.normal(self.mu_n, self.sigma_n), -1, 0)
            
            
def main():
    # load the states, next states, rewards from a csv / log file
    # use phase2-selected/subject_zhihan/additional_log_files/aggregated_states_v2_all_acc100.csv, use key pre_state,state, gt as the states, next states, rewards
    # reward is in [-1, 1]
    # add a configurable acc option and a configurable negative_only option
    # if negative_only is True, then the rewards are in {0, -1}, and the reward is 
    # only accurate acc percentage of the time (which means that 1-acc percentage of the time, the reward is the opposite of the actual reward 0 <-> -1)
    # if negative_only is False, then the rewards are in {-1, 0, 1}, and the reward is accurate acc percentage of the time, and the rest are random incorrect rewards
    # and call check_feedback_processor
    import argparse, csv, ast

    parser = argparse.ArgumentParser(description="Replay CSV, synthesize noisy rewards, and visualize feedback processor.")
    parser.add_argument("--csv", default=str(Path("phase2-selected/subject_zhihan/additional_log_files/aggregated_states_v2_all_acc100.csv")), help="Path to CSV with columns pre_state,state,gt")
    parser.add_argument("--acc", type=float, default=1.0, help="Accuracy probability for noisy reward generation [0,1]")
    parser.add_argument("--negative-only", action="store_true", help="Use reward set {0,-1} instead of {-1,0,1}")
    parser.add_argument("--processor", choices=["gp", "gp_uq", "tiny"], default="gp", help="Feedback processor backend")
    parser.add_argument("--max-rows", type=int, default=500, help="Limit number of rows processed from CSV")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for noise")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rng = np.random.default_rng(args.seed)

    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    rewards_gt: list[int] = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # csv module decodes doubled quotes; ast.literal_eval parses JSON-like list of strings
            pre_state_list = ast.literal_eval(row["pre_state"])  # list[str]
            state_list = ast.literal_eval(row["state"])          # list[str]
            s = parse_grid(pre_state_list)
            ns = parse_grid(state_list)
            gt_val = int(row.get("gt", 0))
            if args.negative_only and gt_val == 1:
                gt_val = 0
            states.append(s)
            next_states.append(ns)
            rewards_gt.append(gt_val)
            if args.max_rows is not None and len(states) >= args.max_rows:
                break

    def corrupt_reward(true_r: int) -> int:
        if rng.random() <= args.acc:
            return int(true_r)
        if args.negative_only:
            return 0 if true_r == -1 else -1
        # choose randomly among incorrect labels
        candidates = [-1, 0, 1]
        if true_r in candidates:
            candidates.remove(int(true_r))
        return int(rng.choice(candidates))

    noisy_rewards = [corrupt_reward(r) for r in rewards_gt]

    # Choose feedback processor
    if args.processor == "gp":
        fp = GPPFeedbackPreProcessor(feature_version="v4", negative_feedback_only=args.negative_only)
    elif args.processor == "gp_uq":
        fp = GPPUQFeedbackPreProcessor(feature_version="v4", negative_feedback_only=args.negative_only)
    else:
        fp = TINYMLFeedbackPreProcessor(feature_version="v4", negative_feedback_only=args.negative_only)

    # Run visualization checkpoints
    check_feedback_processor(states, next_states, noisy_rewards, fp)

    print(f"Processed {len(states)} transitions. Visualizations saved under ./feedback_v4_previews")

if __name__ == "__main__":
    main()