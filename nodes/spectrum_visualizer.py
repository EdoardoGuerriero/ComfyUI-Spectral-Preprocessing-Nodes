"""
Node 6 — FFT Spectrum Visualizer (Debug)
==========================================
Renders diagnostic views of the FFT spectrum of an input image.

Outputs a single IMAGE tensor containing a 2×3 grid of panels:

  ┌──────────────────┬──────────────────┬──────────────────┐
  │  Log Magnitude   │     Phase        │  Radial Profile  │
  ├──────────────────┼──────────────────┼──────────────────┤
  │ Angular Profile  │  Detected Spikes │   Gain Mask      │
  └──────────────────┴──────────────────┴──────────────────┘

Panel descriptions:
  Log Magnitude   — log(1 + |FFT|) normalised to [0,1], DC centred.
                    Bright spots = concentrated energy.
  Phase           — FFT phase angle mapped to [0,1].  Uniform = natural image.
  Radial Profile  — Plot of mean magnitude vs normalised radius, with a
                    reference 1/f^2 curve overlaid in a different colour.
  Angular Profile — Plot of summed magnitude energy vs angle (0°–180°).
                    Spikes = directional artifacts.
  Detected Spikes — Binary/heatmap mask of pixels flagged as anomalous spikes
                    by the same z-score method used in Node 1.
  Gain Mask       — The actual gain map that would be applied by Node 1 at the
                    given parameters, displayed as a heatmap (white = no change,
                    dark = suppressed).

All panels are rendered in greyscale and packed into an RGB image for
compatibility with ComfyUI preview nodes.  The input image is treated as
the *average of its RGB channels* for the spectrum analysis so the display
remains single-channel and unambiguous.

The `channel` parameter lets you visualise a specific channel (R/G/B/avg)
instead of the average.
"""

import numpy as np
from scipy.ndimage import median_filter

from ..utils.fft_utils import (
    tensor_to_numpy,
    numpy_to_tensor,
    fft2_channel,
    magnitude,
    phase,
    radial_grid,
    angular_grid,
    estimate_background_median,
    stable_sigmoid,
    get_dc_mask,
    clamp01,
)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _norm01(arr: np.ndarray) -> np.ndarray:
    """Normalise an array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _make_blank(h: int, w: int) -> np.ndarray:
    """Return a grey [H, W] panel."""
    return np.full((h, w), 0.15, dtype=np.float32)


def _draw_curve(panel: np.ndarray, xs: np.ndarray, ys: np.ndarray, val: float = 1.0, width: int = 2) -> None:
    """
    Draw a 1-D curve on a greyscale panel in-place.

    xs : x positions in [0, 1] (normalised to panel width)
    ys : y positions in [0, 1] (normalised to panel height, 0 = bottom)
    val: greyscale value of the drawn line
    """
    h, w = panel.shape
    px = np.clip((xs * (w - 1)).astype(int), 0, w - 1)
    py = np.clip(((1 - ys) * (h - 1)).astype(int), 0, h - 1)
    for i in range(len(px)):
        for dw in range(-width // 2, width // 2 + 1):
            c = np.clip(px[i] + dw, 0, w - 1)
            panel[py[i], c] = val


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def _panel_log_magnitude(spectrum: np.ndarray, h: int, w: int) -> np.ndarray:
    mag    = magnitude(spectrum)
    log_m  = np.log1p(mag)
    return _norm01(log_m).astype(np.float32)


def _panel_phase(spectrum: np.ndarray, h: int, w: int) -> np.ndarray:
    ph = phase(spectrum)
    return _norm01(ph).astype(np.float32)


def _panel_radial_profile(
    spectrum: np.ndarray, r_norm: np.ndarray, h: int, w: int, panel_h: int, panel_w: int
) -> np.ndarray:
    mag      = magnitude(spectrum)
    n_bins   = 128
    edges    = np.linspace(0.0, 1.0, n_bins + 1)
    centers  = 0.5 * (edges[:-1] + edges[1:])
    means    = np.array([
        mag[(r_norm >= edges[i]) & (r_norm < edges[i + 1])].mean()
        if ((r_norm >= edges[i]) & (r_norm < edges[i + 1])).any() else 0.0
        for i in range(n_bins)
    ])

    panel = _make_blank(panel_h, panel_w)

    # Measured profile (white)
    safe_means = np.where(means < 1e-8, 1e-8, means)
    log_means  = np.log(safe_means)
    y_norm     = _norm01(log_means)
    _draw_curve(panel, centers, y_norm, val=1.0)

    # Reference 1/f^2 curve (mid-grey)
    safe_c    = np.where(centers < 1e-4, 1e-4, centers)
    reference = 1.0 / (safe_c ** 2)
    ref_log   = np.log(reference)
    ref_norm  = _norm01(ref_log)
    _draw_curve(panel, centers, ref_norm, val=0.55)

    return panel


def _panel_angular_profile(
    spectrum: np.ndarray, r_norm: np.ndarray, panel_h: int, panel_w: int, min_r: float = 0.05
) -> np.ndarray:
    h_s, w_s = spectrum.shape
    mag      = magnitude(spectrum)
    angles   = angular_grid(h_s, w_s)
    ring     = r_norm >= min_r

    n_bins  = 180
    edges   = np.linspace(0.0, np.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    energy  = np.array([
        mag[(ring) & (angles >= edges[i]) & (angles < edges[i + 1])].sum()
        for i in range(n_bins)
    ])

    panel  = _make_blank(panel_h, panel_w)
    xs     = centers / np.pi        # normalise to [0, 1]
    ys     = _norm01(energy)
    _draw_curve(panel, xs, ys, val=1.0)
    return panel


def _panel_spike_mask(
    spectrum: np.ndarray,
    kernel_size: int,
    threshold_sigma: float,
    panel_h: int,
    panel_w: int,
) -> np.ndarray:
    mag        = magnitude(spectrum)
    background = estimate_background_median(mag, kernel_size)
    residual   = mag - background
    mad        = np.median(np.abs(residual - np.median(residual)))
    std        = mad * 1.4826 + 1e-8
    z          = residual / std
    # Heatmap: z-score clipped and normalised
    heatmap    = _norm01(np.clip(z, 0, threshold_sigma * 3))
    return heatmap.astype(np.float32)


def _panel_gain_mask(
    spectrum: np.ndarray,
    kernel_size: int,
    threshold_sigma: float,
    attenuation: float,
    dc_mask: np.ndarray,
    panel_h: int,
    panel_w: int,
) -> np.ndarray:
    mag        = magnitude(spectrum)
    background = estimate_background_median(mag, kernel_size)
    residual   = mag - background
    mad        = np.median(np.abs(residual - np.median(residual)))
    std        = mad * 1.4826 + 1e-8
    z          = residual / std
    overshoot  = z - threshold_sigma
    sigmoid    = stable_sigmoid(overshoot * 2.0)
    gain       = 1.0 - attenuation * sigmoid
    gain       = np.clip(gain, 1.0 - attenuation, 1.0)
    gain[dc_mask] = 1.0
    return gain.astype(np.float32)


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------

def _assemble_grid(panels: list, rows: int = 2, cols: int = 3) -> np.ndarray:
    """
    Assemble a list of [H, W] greyscale panels into a [rows*H, cols*W, 3] RGB image.
    """
    ph, pw = panels[0].shape
    grid   = np.zeros((rows * ph, cols * pw, 3), dtype=np.float32)

    for idx, panel in enumerate(panels):
        r, c = divmod(idx, cols)
        p_rgb = np.stack([panel, panel, panel], axis=-1)
        grid[r * ph: (r + 1) * ph, c * pw: (c + 1) * pw] = p_rgb

    return grid


# ---------------------------------------------------------------------------
# Main per-image function
# ---------------------------------------------------------------------------

def _visualize_image(
    image: np.ndarray,
    channel: str,
    kernel_size: int,
    threshold_sigma: float,
    attenuation: float,
    panel_size: int,
) -> np.ndarray:
    h, w, c = image.shape
    ph = pw = panel_size

    # Select channel
    if channel == "R":
        img_ch = image[:, :, 0]
    elif channel == "G":
        img_ch = image[:, :, 1]
    elif channel == "B":
        img_ch = image[:, :, 2]
    else:
        img_ch = image.mean(axis=2)

    spectrum = fft2_channel(img_ch)
    r_norm   = radial_grid(h, w)
    dc_mask  = get_dc_mask(h, w, radius=2)

    panels = [
        _panel_log_magnitude(spectrum, h, w),
        _panel_phase(spectrum, h, w),
        _panel_radial_profile(spectrum, r_norm, h, w, ph, pw),
        _panel_angular_profile(spectrum, r_norm, ph, pw),
        _panel_spike_mask(spectrum, kernel_size, threshold_sigma, ph, pw),
        _panel_gain_mask(spectrum, kernel_size, threshold_sigma, attenuation, dc_mask, ph, pw),
    ]

    # All panels must be panel_size × panel_size; resize if needed
    resized = []
    for p in panels:
        if p.shape != (ph, pw):
            # Simple nearest-neighbour resize via slicing
            ry = np.linspace(0, p.shape[0] - 1, ph).astype(int)
            rx = np.linspace(0, p.shape[1] - 1, pw).astype(int)
            resized.append(p[np.ix_(ry, rx)])
        else:
            resized.append(p)

    grid = _assemble_grid(resized, rows=2, cols=3)
    return clamp01(grid)


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class FFTSpectrumVisualizer:
    """
    Debug node: renders a 6-panel diagnostic grid of the FFT spectrum.

    Panels (left→right, top→bottom):
      1. Log Magnitude      — bright spots = concentrated spectral energy
      2. Phase              — uniform texture = natural; structured = artifacts
      3. Radial Profile     — white = measured, grey = 1/f² reference
      4. Angular Profile    — energy vs angle; spikes = directional artifacts
      5. Spike Heatmap      — z-score map from Node 1's detection step
      6. Gain Mask          — what Node 1 would apply at these parameters

    Connect the output to a PreviewImage node to inspect it.
    """

    CATEGORY = "Spectral Preprocessing"

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("visualization",)
    FUNCTION      = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (
                    ["avg", "R", "G", "B"],
                    {
                        "default": "avg",
                        "tooltip": "Which channel to analyse.  'avg' uses the mean of RGB.",
                    },
                ),
                "kernel_size": (
                    "INT",
                    {
                        "default": 15,
                        "min": 3,
                        "max": 63,
                        "step": 2,
                        "tooltip": "Median filter size — match to SpectralSpikeSuppressor.",
                    },
                ),
                "threshold_sigma": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.5,
                        "max": 20.0,
                        "step": 0.1,
                        "display": "slider",
                        "tooltip": "Detection threshold — match to SpectralSpikeSuppressor.",
                    },
                ),
                "attenuation": (
                    "FLOAT",
                    {
                        "default": 0.85,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Attenuation shown in the gain mask panel.",
                    },
                ),
                "panel_size": (
                    "INT",
                    {
                        "default": 256,
                        "min": 128,
                        "max": 512,
                        "step": 64,
                        "tooltip": "Pixel size of each individual panel in the grid.",
                    },
                ),
            }
        }

    def apply(
        self,
        image: "torch.Tensor",
        channel: str,
        kernel_size: int,
        threshold_sigma: float,
        attenuation: float,
        panel_size: int,
    ) -> tuple:
        if kernel_size % 2 == 0:
            kernel_size += 1

        arr = tensor_to_numpy(image)

        # Process only the first image in the batch for the visualizer
        vis = _visualize_image(
            arr[0],
            channel=channel,
            kernel_size=kernel_size,
            threshold_sigma=threshold_sigma,
            attenuation=attenuation,
            panel_size=panel_size,
        )

        # Return as batch of 1
        out = vis[np.newaxis, ...]
        return (numpy_to_tensor(out),)
