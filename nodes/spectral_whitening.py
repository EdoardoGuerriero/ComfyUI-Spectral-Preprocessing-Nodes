"""
Node 16 — Spectral Whitening
==============================
Normalizes the FFT magnitude spectrum by the measured radial power envelope,
suppressing the dominant bias that makes some frequency bands carry far more
energy than others.

Distinction from RadialSpectrumNormalizer (Node 2)
---------------------------------------------------
RadialSpectrumNormalizer fits a parametric 1/f^α model to the observed power
spectrum, then gently nudges the spectrum toward that model.  The shape of the
correction depends on how well the model fits the image.

SpectralWhitening is purely data-driven:
  1.  Measure the actual radial mean power P(r) directly from the image.
  2.  Divide each coefficient by sqrt(P(r)) at its radius.
  3.  Optionally target a specific slope α (flat=white, α=2=pink/natural).

The key practical difference:
  • RadialNorm corrects deviations from a *theoretical* 1/f^α slope.
  • SpectralWhitening corrects the actual measured power bias regardless of
    what shape it has — no model assumption.  If an image has a weird resonance
    that makes the spectrum bump at a specific radial band, SpectralWhitening
    flattens it; RadialNorm would not.

Two modes
---------
  "flatten"   → divide by sqrt(P(r)), target = constant (true whitening).
                 At strength=1 every radial bin has exactly equal mean power.
                 Aggressive — removes all radial energy variation.
                 Good for: pre-processing before algorithms that assume white noise.

  "target_1f" → divide by sqrt(P(r)), multiply by sqrt(r^(−target_alpha)).
                 Result: spectrum shaped like 1/f^target_alpha regardless of
                 what the input spectrum looks like.
                 Good for: matching a natural-image power law after processing
                 artifacts have distorted the slope.

Algorithm (per channel)
-----------------------
  1.  FFT → mag, phase
  2.  Build radial grid r_norm ∈ [0, 1]
  3.  For each radial bin [r_lo, r_hi]:
        P(bin) = mean(mag²) over coefficients in bin
  4.  Smooth P(bin) along the radial axis with a Gaussian to avoid sharp
      per-bin gain steps.
  5.  Per-coefficient normalisation:
        mode "flatten":   gain = 1 / sqrt(P(r))
        mode "target_1f": gain = sqrt(r^(-alpha)) / sqrt(P(r))
      Gains are clamped to [min_gain, max_gain].
  6.  mag_white = mag * gain
  7.  Blend: mag_out = (1−strength)·mag + strength·mag_white
  8.  Protect DC.
  9.  IFFT.

Parameters
----------
  strength      : blend factor (0 = no change, 1 = full whitening)
  mode          : "flatten" or "target_1f"
  target_alpha  : target slope for "target_1f" mode (2.0 = natural pink noise)
  n_radial_bins : radial resolution for power estimation (more = finer, noisier)
  smoothing     : Gaussian sigma for smoothing the radial gain curve (bins)
  min_gain      : clamp gain from below (prevent over-amplifying noise floor)
  max_gain      : clamp gain from above (prevent over-suppressing DC-adjacent)
  preserve_dc   : always keep DC coefficient unchanged
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..utils.fft_utils import (
    tensor_to_numpy,
    numpy_to_tensor,
    fft2_channel,
    ifft2_channel,
    magnitude,
    phase,
    reconstruct,
    radial_grid,
    get_dc_mask,
    process_batch_tiled,
    center_crop_to_match,
    clamp01,
)


# ---------------------------------------------------------------------------
# Core algorithm (single channel)
# ---------------------------------------------------------------------------

def _whiten_channel(
    channel: np.ndarray,
    strength: float,
    mode: str,
    target_alpha: float,
    n_radial_bins: int,
    smoothing: float,
    min_gain: float,
    max_gain: float,
    r_norm: np.ndarray,
    dc_mask: np.ndarray,
) -> np.ndarray:
    spec = fft2_channel(channel)
    mag  = magnitude(spec).astype(np.float64)
    ph   = phase(spec)

    # ---- Radial power estimation ----
    edges = np.linspace(0.0, 1.0, n_radial_bins + 1)

    # Per-radial-bin mean power
    P = np.zeros(n_radial_bins, dtype=np.float64)
    bin_centres = 0.5 * (edges[:-1] + edges[1:])

    for i in range(n_radial_bins):
        mask = (r_norm >= edges[i]) & (r_norm < edges[i + 1])
        if mask.any():
            P[i] = (mag[mask] ** 2).mean()

    # Fill empty bins by interpolation
    nonempty = P > 0
    if nonempty.sum() < 2:
        return ifft2_channel(spec).astype(np.float32)

    P = np.interp(bin_centres, bin_centres[nonempty], P[nonempty])
    P = np.maximum(P, 1e-20)  # avoid log(0) / div-by-zero

    # Smooth the power curve
    if smoothing > 0:
        sigma = max(0.5, smoothing * n_radial_bins)
        P = gaussian_filter1d(P, sigma=sigma, mode="nearest")
        P = np.maximum(P, 1e-20)

    # ---- Build per-bin gain ----
    if mode == "target_1f":
        # Target: power ∝ r^(-alpha)  →  mag_target ∝ r^(-alpha/2)
        # Add small epsilon to avoid r=0 blow-up at DC
        r_eps = np.maximum(bin_centres, 1.0 / n_radial_bins)
        P_target = r_eps ** (-target_alpha)
        # Normalise so the gain is 1 on average (preserve overall level)
        scale = np.sqrt(P.mean() / P_target.mean())
        gain_per_bin = scale * np.sqrt(P_target / P)
    else:  # "flatten"
        # Divide by sqrt(P(r)) to make all bands equal power
        scale = np.sqrt(P.mean())
        gain_per_bin = scale / np.sqrt(P)

    gain_per_bin = np.clip(gain_per_bin, min_gain, max_gain)

    # ---- Build per-pixel gain map ----
    gain_map = np.ones_like(mag, dtype=np.float64)
    for i in range(n_radial_bins):
        mask = (r_norm >= edges[i]) & (r_norm < edges[i + 1])
        if mask.any():
            gain_map[mask] = gain_per_bin[i]

    mag_white = mag * gain_map
    mag_out   = (1.0 - strength) * mag + strength * mag_white

    mag_out[dc_mask] = mag[dc_mask]

    return ifft2_channel(reconstruct(mag_out, ph))


# ---------------------------------------------------------------------------
# Per-image wrapper
# ---------------------------------------------------------------------------

def _whiten_image(
    image: np.ndarray,
    strength: float,
    mode: str,
    target_alpha: float,
    n_radial_bins: int,
    smoothing: float,
    min_gain: float,
    max_gain: float,
) -> np.ndarray:
    h, w, c = image.shape
    r_norm  = radial_grid(h, w)
    dc_mask = get_dc_mask(h, w, radius=2)

    out = np.empty_like(image)
    for ch in range(c):
        out[:, :, ch] = _whiten_channel(
            image[:, :, ch],
            strength=strength,
            mode=mode,
            target_alpha=target_alpha,
            n_radial_bins=n_radial_bins,
            smoothing=smoothing,
            min_gain=min_gain,
            max_gain=max_gain,
            r_norm=r_norm,
            dc_mask=dc_mask,
        )
    return out


def _whiten_image_fn(image: np.ndarray, **kwargs) -> np.ndarray:
    return _whiten_image(image, **kwargs)


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class SpectralWhitening:
    """
    Normalizes the FFT spectrum by the measured radial power envelope.

    Unlike RadialSpectrumNormalizer (which fits a 1/f^α model), this node
    is fully data-driven — it divides each coefficient by the ACTUAL measured
    power at its radial frequency, with no model assumption.  This corrects
    any radial bias, regardless of its shape.

    Two modes:
      • "flatten"   — true whitening: all radial bands get equal mean power.
      • "target_1f" — data-driven reshape to target_alpha slope (default 2.0
                       = natural pink noise).  More aggressive than
                       RadialSpectrumNormalizer because it measures and corrects
                       the actual bias rather than assuming a model shape.

    Phase is never modified.  The DC coefficient is always protected.
    """

    CATEGORY = "Spectral Preprocessing"

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "0 = no change, 1 = full whitening.",
                    },
                ),
                "mode": (
                    ["target_1f", "flatten"],
                    {
                        "default": "target_1f",
                        "tooltip": (
                            "'target_1f' — reshape power spectrum to r^(−target_alpha).  "
                            "'flatten'   — true whitening: equal power at all radii (aggressive)."
                        ),
                    },
                ),
                "target_alpha": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": (
                            "Target power spectral slope for 'target_1f' mode.  "
                            "2.0 = natural pink noise (typical natural images).  "
                            "0.0 = flat / white (same as 'flatten' mode).  "
                            "Values above 2 over-emphasise low frequencies.  "
                            "Ignored in 'flatten' mode."
                        ),
                    },
                ),
                "n_radial_bins": (
                    "INT",
                    {
                        "default": 64,
                        "min": 8,
                        "max": 256,
                        "step": 8,
                        "tooltip": (
                            "Number of radial bins for power estimation.  More bins = finer "
                            "correction but more sensitive to per-bin noise.  64 is a good default."
                        ),
                    },
                ),
                "smoothing": (
                    "FLOAT",
                    {
                        "default": 0.08,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Gaussian smoothing sigma for the radial gain curve, as a fraction "
                            "of n_radial_bins.  Higher values prevent per-bin ringing artifacts.  "
                            "0.05–0.15 recommended."
                        ),
                    },
                ),
                "min_gain": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.001,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Minimum per-bin gain — prevents amplifying the noise floor beyond this.",
                    },
                ),
                "max_gain": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 1.0,
                        "max": 200.0,
                        "step": 1.0,
                        "tooltip": "Maximum per-bin gain — prevents over-suppressing near-DC bands.",
                    },
                ),
                "tile_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Tile size for large images. 0 = whole image (recommended).",
                    },
                ),
                "tile_overlap": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 512,
                        "step": 32,
                        "tooltip": "Tile overlap in pixels.",
                    },
                ),
            }
        }

    def apply(
        self,
        image: "torch.Tensor",
        strength: float,
        mode: str,
        target_alpha: float,
        n_radial_bins: int,
        smoothing: float,
        min_gain: float,
        max_gain: float,
        tile_size: int,
        tile_overlap: int,
    ) -> tuple:
        arr            = tensor_to_numpy(image)
        orig_h, orig_w = arr.shape[1], arr.shape[2]

        kwargs = dict(
            strength=strength,
            mode=mode,
            target_alpha=target_alpha,
            n_radial_bins=n_radial_bins,
            smoothing=smoothing,
            min_gain=min_gain,
            max_gain=max_gain,
        )

        out = process_batch_tiled(arr, _whiten_image_fn, tile_size, tile_overlap, **kwargs)
        out = center_crop_to_match(out, orig_h, orig_w)
        return (numpy_to_tensor(clamp01(out)),)
