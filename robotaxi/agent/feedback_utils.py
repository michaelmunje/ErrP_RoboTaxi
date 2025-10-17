from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import sys
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap
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
from robotaxi.agent.game_feature_utils import compute_delta_features_v4, compute_delta_features_v6

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

class NonBlockingPlotter:
    """Singleton plot manager to reuse a single window and show non-blocking.

    - Use get_figure(key, nrows, ncols, ...) to obtain a persistent figure/axes.
    - Use show(fig) for non-blocking refresh; next show updates the same window.
    - Use close(key) to close and release a managed figure.
    """
    _instance = None

    def __init__(self):
        self._store = {}
        try:
            plt.ion()
        except Exception:
            pass

    @classmethod
    def instance(cls) -> "NonBlockingPlotter":
        if cls._instance is None:
            cls._instance = NonBlockingPlotter()
        return cls._instance

    def get_figure(self, key: str, nrows: int = 1, ncols: int = 1,
                   figsize: Tuple[float, float] = (3, 2.5), dpi: int = 120,
                   constrained_layout: bool = False):
        entry = self._store.get(key)
        if entry is None or not plt.fignum_exists(entry["fig"].number):
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, constrained_layout=constrained_layout)
            self._store[key] = {"fig": fig, "shape": (nrows, ncols)}
            return fig, axes
        fig = entry["fig"]
        fig.clf()
        fig.set_constrained_layout(bool(constrained_layout))
        axes = fig.subplots(nrows, ncols)
        self._store[key]["shape"] = (nrows, ncols)
        return fig, axes

    def show(self, fig, pause: float = 0.001) -> None:
        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            pass
        try:
            plt.show(block=False)
        except TypeError:
            plt.show()
        try:
            plt.pause(pause)
        except Exception:
            pass

    def close(self, key: str) -> None:
        entry = self._store.pop(key, None)
        if entry is not None:
            try:
                plt.close(entry["fig"])
            except Exception:
                pass

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
        assert feature_version in ["v4", "v6"], "Invalid feature version"
        self.feature_version = feature_version
        self.negative_feedback_only = negative_feedback_only
        if feature_version == "v4":
            self.featurize_fn = lambda state, next_state: compute_delta_features_v4(state, next_state)[2:4]
        elif feature_version == "v6":
            self.featurize_fn = lambda state, next_state: compute_delta_features_v6(state, next_state)[2:4]
        else:
            raise ValueError(f"Invalid feature version: {feature_version}")

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
        if self.feature_version not in ["v4", "v6"]:
            raise ValueError("visualize_current_processor only supports feature_version='v4' or 'v6'")

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

        # Plot (reuse window)
        nbp = NonBlockingPlotter.instance()
        if self.feature_version == "v4":
            fig, ax = nbp.get_figure("feedback_v4_gp", nrows=1, ncols=1, figsize=(3, 2.5), dpi=120)
        elif self.feature_version == "v6":
            fig, ax = nbp.get_figure("feedback_v6_gp", nrows=1, ncols=1, figsize=(3, 2.5), dpi=120)
        else:
            raise ValueError(f"Invalid feature version: {self.feature_version}")
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
        if self.feature_version == "v4":
            ax.set_xlabel("X1 (v4 Δsum(1/passenger) proximity)"); ax.set_ylabel("X2 (v4 Δsum(1/obstacle) proximity)")
        elif self.feature_version == "v6":
            ax.set_xlabel("X1 (v6 Δmin(passenger) proximity)"); ax.set_ylabel("X2 (v6 Δmin(obstacle) proximity)")
        else:
            raise ValueError(f"Invalid feature version: {self.feature_version}")
        ax.set_title("GPPFeedbackPreProcessor — current adjusted feedback")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show_plot:
            nbp.show(fig)
        # do not close: keep same window for next update

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

        # Plot side-by-side: mean | uncertainty (reuse window)
        nbp = NonBlockingPlotter.instance()
        fig, axes = nbp.get_figure("feedback_v4_gp_uq", nrows=1, ncols=2, figsize=(5.5, 2.5), dpi=120, constrained_layout=True)

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
            nbp.show(fig)
        # keep window open for reuse

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
        weight_decay: float = 0, # used to be 2e-4,
        threshold: float = 0.4
    ):
        self.threshold = float(threshold)
        assert feature_version in ["v4", "v6"], "Invalid feature version"
        print(f"feature_version: {feature_version}")
        self.feature_version = feature_version
        self.negative_feedback_only = negative_feedback_only
        if feature_version == "v4":
            self.featurize_fn = lambda state, next_state: compute_delta_features_v4(state, next_state)[2:4]
        elif feature_version == "v6":
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
        
        # create a dummy visualization window that shows the current processor
        nbp = NonBlockingPlotter.instance()
        fig, ax = nbp.get_figure(
            f"count_based_{self.feature_version}_heatmap",
            nrows=1, ncols=1, figsize=(3, 2.5), dpi=120
        )
        nbp.show(fig)

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
        xf = np.asarray(self.featurize_fn(state, next_state), dtype=float).reshape(-1)
        z = self._predict(xf)
        return float(clip_output(z, self.negative_feedback_only))
    
    def visualize_current_processor(self, save_path: str = "", show_plot: bool = False) -> None:
        if self.feature_version not in ["v4", "v6"]:
            raise ValueError("visualize_current_processor only supports feature_version='v4' or 'v6'")

        # Grid over [-2, 2]^2 for v4, [-10, 10]^2 for v6
        g = 60
        xmin, xmax, ymin, ymax = (-2.0, 2.0, -2.0, 2.0) 
        gx = np.linspace(xmin, xmax, g)
        gy = np.linspace(ymin, ymax, g)
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

        # Plot (reuse window)
        nbp = NonBlockingPlotter.instance()
        if self.feature_version == "v4":
            fig, ax = nbp.get_figure("feedback_v4_tinyml", nrows=1, ncols=1, figsize=(3, 2.5), dpi=120)
        elif self.feature_version == "v6":
            fig, ax = nbp.get_figure("feedback_v6_tinyml", nrows=1, ncols=1, figsize=(3, 2.5), dpi=120)
        else:
            raise ValueError(f"Invalid feature version: {self.feature_version}")
        
        vmin, vmax = -1.0, (0.0 if self.negative_feedback_only else 1.0)
        if USE_LINEAR_COLOR:
            im = ax.imshow(Z, extent=(xmin, xmax, ymin, ymax), origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
        else:
            norm = SymLogNorm(linthresh=SYMLIN_COLOR_LINTHRESH, vmin=vmin, vmax=vmax)
            im = ax.imshow(Z, extent=(xmin, xmax, ymin, ymax), origin="lower", norm=norm, interpolation="nearest")
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
        if self.feature_version == "v4":
            ax.set_xlabel("X1 (v4 Δsum(1/passenger) proximity)"); ax.set_ylabel("X2 (v4 Δsum(1/obstacle) proximity)")
        elif self.feature_version == "v6":
            ax.set_xlabel("X1 (v6 Δmin(passenger) proximity)"); ax.set_ylabel("X2 (v6 Δmin(obstacle) proximity)")
        else:
            raise ValueError(f"Invalid feature version: {self.feature_version}")
        
        ax.set_title("TINYMLFeedbackPreProcessor — current adjusted feedback")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show_plot:
            nbp.show(fig)
        # keep window open for reuse
        
class TinyBernoulliFeedbackPreProcessor(visualizeableFeedbackPreProcessorV4):
    """
    One-layer logistic regressor over features to model P(y=0|x).
    Outputs:
      mean  = -1 + p
      var   = p * (1 - p)
    Trained online with BCE-with-logits + Adam + L2.
    """
    def __init__(
        self,
        feature_version: str = "v4",
        negative_feedback_only: bool = True,   # targets are {-1, 0}
        seed: int = 123,
        lr: float = 3e-3,
        steps_per_update: int = 100,
        batch_size: int = 512,                 # fine w/ <1000 data
        weight_decay: float = 2e-4
    ):
        assert feature_version in ["v4", "v6"], "Invalid feature version"
        print(f"feature_version: {feature_version}")
        self.feature_version = feature_version
        self.negative_feedback_only = negative_feedback_only

        if feature_version == "v4":
            self.featurize_fn = lambda state, next_state: compute_delta_features_v4(state, next_state)[2:4]
        else:
            self.featurize_fn = lambda state, next_state: compute_delta_features_v6(state, next_state)[2:4]

        # Replay
        self._X: List[np.ndarray] = []
        self._y: List[float] = []

        # Opt/Train
        self.rng = np.random.default_rng(seed)
        self.lr = float(lr)
        self.steps = int(steps_per_update)
        self.batch = int(batch_size)
        self.wd = float(weight_decay)

        # Params (lazy)
        self._W = None   # [d_in, 1]
        self._b = None   # [1]
        # Adam buffers
        self._mW = self._vW = self._mb = self._vb = None
        self._t = 0

    # ---------- utils ----------
    @staticmethod
    def _sigmoid(z):
        # stable sigmoid
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def _labels_ok(self, y: float) -> float:
        # restrict to {-1, 0}
        return 0.0 if y >= -0.5 else -1.0

    def _ensure_params(self, d_in: int):
        if self._W is not None: return
        rs = self.rng
        self._W = rs.normal(0, 1/np.sqrt(d_in), size=(d_in, 1))
        self._b = np.zeros(1)
        self._mW = np.zeros_like(self._W); self._vW = np.zeros_like(self._W)
        self._mb = np.zeros_like(self._b); self._vb = np.zeros_like(self._b)
        self._t = 0

    def _forward_logits(self, X: np.ndarray) -> np.ndarray:
        # X: [N, d], returns logits [N, 1]
        return X @ self._W + self._b

    def _adam_step(self, gW: np.ndarray, gb: np.ndarray, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        self._mW = b1*self._mW + (1-b1)*gW
        self._vW = b2*self._vW + (1-b2)*(gW*gW)
        self._mb  = b1*self._mb  + (1-b1)*gb
        self._vb  = b2*self._vb  + (1-b2)*(gb*gb)

        mW_hat = self._mW / (1 - b1**self._t)
        vW_hat = self._vW / (1 - b2**self._t)
        mb_hat = self._mb  / (1 - b1**self._t)
        vb_hat = self._vb  / (1 - b2**self._t)

        self._W -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self._b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def _train_few_steps(self):
        if len(self._X) == 0: return
        X = np.vstack(self._X)                         # [N, d]
        y = np.asarray(self._y, dtype=float)           # [-1 or 0], [N]
        self._ensure_params(X.shape[1])

        # map y∈{-1,0} -> t∈{0,1} with t = 1 if y==0 else 0
        t = (y == 0.0).astype(np.float64).reshape(-1, 1)

        N = X.shape[0]
        bs = min(self.batch, N)

        for _ in range(self.steps):
            idx = self.rng.choice(N, size=bs, replace=False)
            Xb = X[idx]                 # [bs, d]
            tb = t[idx]                 # [bs, 1]

            logits = self._forward_logits(Xb)          # [bs, 1]
            p = self._sigmoid(logits)                  # [bs, 1]

            # BCE-with-logits gradient wrt logits is (p - t)
            grad_s = (p - tb) / bs                     # [bs, 1], mean over batch

            # L2 regularization on W
            gW = Xb.T @ grad_s + self.wd * self._W     # [d,1]
            gb = grad_s.sum(axis=0)                    # [1]

            self._adam_step(gW, gb)

    def _predict_stats(self, xf: np.ndarray) -> Tuple[float, float, float]:
        """
        Returns (p, mean, var) for a single feature vector.
          p    = P(y=0|x)
          mean = -1 + p  in [-1,0]
          var  = p*(1-p)
        """
        if self._W is None or len(self._X) == 0:
            p = 0.5  # neutral if untrained
            return p, -1 + p, p * (1 - p)
        s = float(xf @ self._W + self._b)
        p = 1.0 / (1.0 + np.exp(-s))
        mean = -1.0 + p
        var = p * (1.0 - p)
        return p, mean, var

    # ---------- public API ----------
    def add_feedback(self, state, action, next_state, noisy_reward) -> None:
        xf = np.asarray(self.featurize_fn(state, next_state), dtype=float).reshape(1, -1)
        y  = self._labels_ok(float(noisy_reward))
        self._X.append(xf.reshape(-1))
        self._y.append(y)
        self._train_few_steps()

    def get_feedback(self, state, action, next_state, noisy_reward) -> float:
        xf = np.asarray(self.featurize_fn(state, next_state), dtype=float).reshape(-1)
        _, mean, _ = self._predict_stats(xf)
        # mean already in [-1,0]; reuse clipper for consistency with your stack
        return float(clip_output(mean, self.negative_feedback_only))

    def visualize_current_processor(self, save_path: str = "", show_plot: bool = False) -> None:
        if self.feature_version not in ["v4", "v6"]:
            raise ValueError("visualize_current_processor only supports feature_version='v4' or 'v6'")

        # Grid over [-2,2]^2  (your request)
        g = 60
        xmin, xmax, ymin, ymax = (-2.0, 2.0, -2.0, 2.0)
        gx = np.linspace(xmin, xmax, g)
        gy = np.linspace(ymin, ymax, g)
        GX, GY = np.meshgrid(gx, gy)
        grid2 = np.stack([GX.ravel(), GY.ravel()], axis=-1)

        # Embed into current feature space (like your v4/v6 embedding rule)
        if self._W is None or len(self._X) == 0:
            Z_mean = np.zeros((g, g))
            Z_var = np.zeros((g, g))
            d_in = 4  # default for embedding layout before init
        else:
            d_in = self._W.shape[0]

        def embed(pt2):
            full = np.zeros(d_in, dtype=float)
            if d_in >= 4:
                full[2] = pt2[0]
                full[3] = pt2[1]
            else:
                full[:min(2, d_in)] = pt2[:min(2, d_in)]
            return full

        if self._W is not None and len(self._X) > 0:
            means = []
            vars_ = []
            for pt in grid2:
                p, m, v = self._predict_stats(embed(pt))
                means.append(m)
                vars_.append(v)
            Z_mean = np.array(means).reshape(g, g)
            Z_mean = clip_output(Z_mean, self.negative_feedback_only)
            Z_var = np.array(vars_).reshape(g, g)  # already in [0, 0.25]

        # Plot side-by-side: mean and variance
        nbp = NonBlockingPlotter.instance()
        fig, axes = nbp.get_figure(
            f"feedback_{self.feature_version}_tinybern_mean_var",
            nrows=1, ncols=2, figsize=(5.5, 2.5), dpi=120
        )

        # Left: mean ([-1,0])
        ax0 = axes[0]
        vmin, vmax = -1.0, 0.0
        if USE_LINEAR_COLOR:
            im0 = ax0.imshow(Z_mean, extent=(xmin, xmax, ymin, ymax), origin="lower",
                             vmin=vmin, vmax=vmax, interpolation="nearest")
        else:
            norm0 = SymLogNorm(linthresh=SYMLIN_COLOR_LINTHRESH, vmin=vmin, vmax=vmax)
            im0 = ax0.imshow(Z_mean, extent=(xmin, xmax, ymin, ymax), origin="lower",
                             norm=norm0, interpolation="nearest")
        c0 = plt.colorbar(im0, ax=ax0); c0.set_label("Predictive mean (-1 to 0)")
        ax0.set_title("Predictive Mean")
        _apply_axis_scale(ax0)
        ax0.set_xlim(-PLOT_ZOOM_LIMIT, PLOT_ZOOM_LIMIT); ax0.set_ylim(-PLOT_ZOOM_LIMIT, PLOT_ZOOM_LIMIT)
        if self.feature_version == "v4":
            ax0.set_xlabel("X1 (v4 Δsum(1/passenger))"); ax0.set_ylabel("X2 (v4 Δsum(1/obstacle))")
        else:
            ax0.set_xlabel("X1 (v6 Δmin(passenger))"); ax0.set_ylabel("X2 (v6 Δmin(obstacle))")

        # Right: variance ([0, 0.25])
        ax1 = axes[1]
        im1 = ax1.imshow(Z_var, extent=(xmin, xmax, ymin, ymax), origin="lower",
                         vmin=0.0, vmax=0.25, interpolation="nearest")
        c1 = plt.colorbar(im1, ax=ax1); c1.set_label("Predictive variance p(1-p)")
        ax1.set_title("Predictive Variance")
        _apply_axis_scale(ax1)
        ax1.set_xlim(-PLOT_ZOOM_LIMIT, PLOT_ZOOM_LIMIT); ax1.set_ylim(-PLOT_ZOOM_LIMIT, PLOT_ZOOM_LIMIT)
        if self.feature_version == "v4":
            ax1.set_xlabel("X1 (v4 Δsum(1/passenger))"); ax1.set_ylabel("X2 (v4 Δsum(1/obstacle))")
        else:
            ax1.set_xlabel("X1 (v6 Δmin(passenger))"); ax1.set_ylabel("X2 (v6 Δmin(obstacle))")

        # Overlay observed points once (on both panes)
        if len(self._X) > 0:
            Xobs = np.vstack(self._X)
            yobs = np.asarray(self._y, dtype=float).astype(int)
            for ax in (ax0, ax1):
                for val, mk in [(0, 'o'), (-1, '_')]:
                    sel = (yobs == val)
                    if np.any(sel):
                        xi = 2 if Xobs.shape[1] > 2 else 0
                        yi = 3 if Xobs.shape[1] > 3 else (1 if Xobs.shape[1] > 1 else 0)
                        ax.scatter(Xobs[sel, xi], Xobs[sel, yi], s=28, marker=mk, alpha=0.9, linewidths=1.2)

        fig.suptitle("Tiny Bernoulli Feedback — Mean & Variance", y=1.02)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show_plot:
            nbp.show(fig)
        # keep window open for reuse
        
        
class CountBasedFeedbackPreProcessor(visualizeableFeedbackPreProcessorV4):
    """
    Count-based local vs global comparator on labels {-1, 0}.
    Maintains 2D features X2 in [-2,2]^2 and noisy labels Y in {-1,0}.
    
    Predict(x):
      A = count(y == -1 for y in Y)
      B = count(y == 0  for y in Y)
      C = count(y == -1 for (x',y) in zip(X2,Y) if |x'-x| <= radius (per-axis))
      D = count(y == 0  for (x',y) in zip(X2,Y) if |x'-x| <= radius (per-axis))

      if C + D < min_local or A + B == 0: return 0
      else return -1 * ( C/(C+D) - A/(A+B) )   # clipped downstream if needed

    Visualize: heatmap of -C/(C+D) on [-2,2]^2.
    """
    def __init__(
        self,
        feature_version: str = "v4",
        negative_feedback_only: bool = True,   # outputs usually in [-1,0]
        radius: Union[float, str] = 'quadrants',                   # "±1 inclusive", tunable, or 'quadrants'
        min_local: int = 10,                    # threshold for local evidence
        seed: int = 123,
        margin: float = 0.03
    ):
        assert feature_version in ["v4", "v6"], "Invalid feature version"
        print(f"feature_version: {feature_version}")
        self.feature_version = feature_version
        self.negative_feedback_only = negative_feedback_only

        if feature_version == "v4":
            self.featurize_fn = lambda state, next_state: compute_delta_features_v4(state, next_state)[2:4]
        else:
            self.featurize_fn = lambda state, next_state: compute_delta_features_v6(state, next_state)[2:4]
        
        self.radius = radius
        if radius != 'quadrants':
            self.radius = float(radius)
        self.min_local = int(min_local)
        self.rng = np.random.default_rng(seed)
        self.margin = float(margin)

        # Replay (store ONLY the 2D features used for locality)
        self._X2: List[np.ndarray] = []
        self._Y:  List[int] = []   # -1 or 0
        
        # create a dummy visualization window that shows the current processor
        nbp = NonBlockingPlotter.instance()
        fig, ax = nbp.get_figure(
            f"count_based_{self.feature_version}_heatmap",
            nrows=1, ncols=1, figsize=(3, 2.5), dpi=120
        )
        nbp.show(fig)

    # ---------- helpers ----------
    @staticmethod
    def _clean_label(y: float) -> int:
        # Snap to {-1,0}
        return 0 if y >= -0.5 else -1

    def _extract_2d(self, state, next_state) -> np.ndarray:
        # Ensure 2D np.array
        vf = np.asarray(self.featurize_fn(state, next_state), dtype=float)
        assert vf.shape[0] == 2, "CountBasedFeedbackPreProcessor only supports 2D features"
        return vf
    
    def _counts_global(self) -> Tuple[int, int]:
        if len(self._Y) == 0:
            return 0, 0
        Y = np.asarray(self._Y, dtype=int)
        A = int(np.sum(Y == -1))
        B = int(np.sum(Y == 0))
        return A, B

    def _counts_local(self, x2: np.ndarray) -> Tuple[int, int, int]:
        """
        Returns (C, D, Nloc) with axis-aligned box |x-x'| <= radius (per coord).
        """
        if len(self._X2) == 0:
            return 0, 0, 0
        # Filter out any invalid/None entries that may have been added before fix
        valid_idx = [i for i, xi in enumerate(self._X2) if isinstance(xi, np.ndarray) and xi.shape == (2,)]
        if len(valid_idx) == 0:
            return 0, 0, 0
        X2 = np.vstack([self._X2[i] for i in valid_idx])  # [N,2]
        Y  = np.asarray([self._Y[i] for i in valid_idx], dtype=int)  # [N]
        # axis-aligned "chebyshev" box: per-axis threshold
        if self.radius == 'quadrants':
            # Local iff each coordinate of X2 is on the same side as x2.
            # Zero is treated as both positive and negative (wildcard).
            s_x = np.sign(x2)      # shape: (d,)
            s_X = np.sign(X2)      # shape: (n, d)

            # A coordinate matches if signs are equal OR either side is zero
            same_side = (s_X == s_x) | (s_x == 0) | (s_X == 0)
            close_mask = same_side.all(axis=1)
        else:
            close_mask = (np.abs(X2 - x2[None, :]) <= self.radius).all(axis=1)
        Nloc = int(np.sum(close_mask))
        if Nloc == 0:
            return 0, 0, 0
        Yloc = Y[close_mask]
        C = int(np.sum(Yloc == -1))
        D = int(np.sum(Yloc == 0))
        return C, D, Nloc

    # ---------- public API ----------
    def add_feedback(self, state, action, next_state, noisy_reward) -> None:
        # if either of state, action, next_state is None, then return
        if state is None or action is None or next_state is None:
            print(f"state, action, next_state is None, returning")
            return
        x2 = self._extract_2d(state, next_state)
        if x2 is None or not isinstance(x2, np.ndarray) or x2.shape != (2,):
            print(f"x2 is None or not isinstance(x2, np.ndarray) or x2.shape != (2,), returning")
            # safety guard
            return
        y  = self._clean_label(float(noisy_reward))
        self._X2.append(x2)
        self._Y.append(y)

    def get_feedback(self, state, action, next_state, noisy_reward) -> float:
        x2 = self._extract_2d(state, next_state)
        A, B = self._counts_global()
        C, D, Nloc = self._counts_local(x2)

        # Rules
        if (C + D) < self.min_local or (A + B) == 0:
            out = noisy_reward
        else:
            p_local_minus1  = C / (C + D)
            p_global_minus1 = A / (A + B)
            if (p_local_minus1 - p_global_minus1) > self.margin:
                out = -1.0
            else:
                out = 0.0
            fnr = 0.3
            # if p_local_minus1 > fnr:
            #     out = noisy_reward
            
            # out = -1.0 * (p_local_minus1 - p_global_minus1)

        # Keep consistent with your stack's clipping convention
        return float(clip_output(out, self.negative_feedback_only))

    def visualize_current_processor(self, save_path: str = "", show_plot: bool = False) -> None:
        if self.feature_version not in ["v4", "v6"]:
            raise ValueError("visualize_current_processor only supports feature_version='v4' or 'v6'")

        # Grid over [-2,2]^2
        g = 60
        xmin, xmax, ymin, ymax = -2.0, 2.0, -2.0, 2.0
        gx = np.linspace(xmin, xmax, g)
        gy = np.linspace(ymin, ymax, g)
        GX, GY = np.meshgrid(gx, gy)
        grid2 = np.stack([GX.ravel(), GY.ravel()], axis=-1)  # [M,2], M=g*g

        # If no data, show zeros
        if len(self._X2) <= 0:
            Z = np.zeros((g, g))
        else:
            X2 = np.vstack(self._X2)             # [N,2]
            Y  = np.asarray(self._Y, dtype=int)  # [N]

            # Vectorized local counts for all grid points:
            # If radius == 'quadrants', local means same sign per coordinate (zeros are wildcards).
            # Else, local means |X2[n]-grid2[m]| <= radius per-axis.
            # We'll broadcast: grid2[M,1,2] vs X2[1,N,2] -> mask[M,N]
            Gm = grid2[:, None, :]                         # [M,1,2]
            Xn = X2[None, :, :]                            # [1,N,2]
            if self.radius == 'quadrants':
                s_G = np.sign(Gm)                          # [M,1,2]
                s_X = np.sign(Xn)                          # [1,N,2]
                same_side = (s_X == s_G) | (s_G == 0) | (s_X == 0)
                close = same_side.all(axis=2)              # [M,N]
            else:
                close = (np.abs(Xn - Gm) <= self.radius).all(axis=2)  # [M,N]

            Yn = Y[None, :]                                # [1,N]
            is_m1 = (Yn == -1)                             # [1,N]
            is_0  = (Yn ==  0)                             # [1,N]

            C_all = (close & is_m1).sum(axis=1)            # [M]
            D_all = (close & is_0 ).sum(axis=1)            # [M]
            denom_local = C_all + D_all                    # [M]

            # Local mean over {-1,0} labels is -C/(C+D); if denom_local==0 -> 0
            local_mean = np.where(denom_local > 0, -C_all / denom_local, 0.0).astype(float)  # [M]

            # Global mean over {-1,0}: -A/(A+B) (scalar)
            A = int(np.sum(Y == -1))
            B = int(np.sum(Y == 0))
            denom_global = A + B
            global_mean = (-A / denom_global) if denom_global > 0 else 0.0

            # Heatmap: (local_mean - global_mean)
            Z_flat = (local_mean - global_mean)
            Z = Z_flat.reshape(g, g)

        # Plot single heatmap per your request
        nbp = NonBlockingPlotter.instance()
        fig, ax = nbp.get_figure(
            f"count_based_{self.feature_version}_heatmap",
            nrows=1, ncols=1, figsize=(3, 2.5), dpi=120
        )

        # Show symmetric range around 0 for local-global difference
        vmin, vmax = -1.0, 1.0
        # Diverging colormap: red (negative) -> white (zero) -> green (positive)
        cmap = LinearSegmentedColormap.from_list("red_white_green", ["red", "white", "green"], N=256)
        if USE_LINEAR_COLOR:
            im = ax.imshow(Z, extent=(xmin, xmax, ymin, ymax), origin="lower",
                           vmin=vmin, vmax=vmax, interpolation="nearest", cmap=cmap)
        else:
            norm = SymLogNorm(linthresh=SYMLIN_COLOR_LINTHRESH, vmin=vmin, vmax=vmax)
            im = ax.imshow(Z, extent=(xmin, xmax, ymin, ymax), origin="lower",
                           norm=norm, interpolation="nearest", cmap=cmap)
        cbar = plt.colorbar(im, ax=ax); cbar.set_label("local mean − global mean")

        # Overlay observed points
        if len(self._X2) > 0:
            Xobs = np.vstack(self._X2)
            yobs = np.asarray(self._Y, dtype=int)
            for val, mk in [(0, 'o'), (-1, '_')]:
                sel = (yobs == val)
                if np.any(sel):
                    ax.scatter(Xobs[sel, 0], Xobs[sel, 1], s=28, marker=mk, alpha=0.9, linewidths=1.2)

        _apply_axis_scale(ax)
        ax.set_xlim(-PLOT_ZOOM_LIMIT, PLOT_ZOOM_LIMIT); ax.set_ylim(-PLOT_ZOOM_LIMIT, PLOT_ZOOM_LIMIT)
        if self.feature_version == "v4":
            ax.set_xlabel("X1 (v4 Δsum(1/passenger))"); ax.set_ylabel("X2 (v4 Δsum(1/obstacle))")
        else:
            ax.set_xlabel("X1 (v6 Δmin(passenger))"); ax.set_ylabel("X2 (v6 Δmin(obstacle))")

        ax.set_title("Count-based Feedback — Heatmap of (local − global) mean")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        if show_plot:
            nbp.show(fig)
        # keep window open




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
        self.tpr = tpr # TP / (TP + FN)
        self.tnr = tnr # TN / (TN + FP)
        
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