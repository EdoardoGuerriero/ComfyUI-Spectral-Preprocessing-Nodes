"""
Node 2 — Radial Spectrum Normalizer
=====================================
Natural photographs follow a well-known statistical regularity:

    Power(f) ∝ 1 / f^α       α ≈ 2  (pink/brown noise in 2-D)

AI-generated images, upscaled images, and heavily compressed images often
deviate from this law — typically carrying excess energy at mid-to-high
frequencies (checkerboards, ringing, texture over-sharpening) or at very
specific radial bands.

This node computes the radial power spectrum, compares it against a target
1/f^α curve, and applies a smooth radial gain function that nudges the image
toward the target.  The correction is blended with a `strength` parameter so
you can apply a partial correction rather than forcing an exact fit.

Algorithm (per channel):
  1.  FFT → magnitude²  (power spectrum)
  2.  Compute radial average of power → P(r)
  3.  Fit / synthesise target curve  T(r) = C / r^α   (C matched to DC)
  4.  Compute radial gain  G(r) = (T(r) / P(r))^0.5   in amplitude space
  5.  Smooth G(r) radially to avoid abrupt transitions
  6.  Optionally freeze G(r) = 1 for the low-frequency band
  7.  Blend:  G_final = 1 + strength * (G − 1)
  8.  Apply gain map to magnitude, reconstruct, IFFT

The gain is applied to amplitude (not power), so squaring is avoided —
this keeps the math equivalent to multiplying the power spectrum by G².
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
    process_batch_tiled,
    center_crop_to_match,
    clamp01,
)


# ---------------------------------------------------------------------------
# Radial statistics helpers
# ---------------------------------------------------------------------------

def _radial_average(power: np.ndarray, r_norm: np.ndarray, n_bins: int) -> tuple:
    """
    Compute the mean power in each radial bin.

    Returns
    -------
    bin_centers : [n_bins] normalised radii (0..1)
    bin_means   : [n_bins] mean power per bin
    """
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means   = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        mask = (r_norm >= bin_edges[i]) & (r_norm < bin_edges[i + 1])
        if mask.any():
            bin_means[i] = power[mask].mean()
        else:
            # Empty bin — interpolate later
            bin_means[i] = np.nan

    # Fill NaN bins by linear interpolation
    nans  = np.isnan(bin_means)
    if nans.any() and not nans.all():
        idx = np.arange(n_bins)
        bin_means[nans] = np.interp(idx[nans], idx[~nans], bin_means[~nans])

    return bin_centers, bin_means


def _build_target_curve(bin_centers: np.ndarray, bin_means: np.ndarray, alpha: float) -> np.ndarray:
    """
    Build a 1/f^alpha target curve scaled to match the input power at low
    frequencies (to preserve overall brightness).

    The DC bin (r≈0) is excluded because 1/f diverges there.
    Scaling anchor: match the mean power of the lowest-frequency non-DC bin.
    """
    # Start from bin index 1 to avoid division by zero at r=0
    anchor_idx = 1
    r_anchor   = bin_centers[anchor_idx]
    p_anchor   = bin_means[anchor_idx]

    # C such that C / r_anchor^alpha = p_anchor
    C = p_anchor * (r_anchor ** alpha)

    # Build target; protect DC bin
    safe_r  = np.where(bin_centers < 1e-6, 1e-6, bin_centers)
    target  = C / (safe_r ** alpha)

    # Cap target at the DC bin's actual power so we never boost DC
    target[0] = bin_means[0]

    return target


def _radial_gain_map(
    r_norm: np.ndarray,
    bin_centers: np.ndarray,
    gain_per_bin: np.ndarray,
) -> np.ndarray:
    """
    Interpolate per-bin gain values onto the full 2-D radial grid.
    Uses linear interpolation — smooth by construction.
    """
    return np.interp(r_norm.ravel(), bin_centers, gain_per_bin).reshape(r_norm.shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Core algorithm (single channel)
# ---------------------------------------------------------------------------

def _normalize_radial_channel(
    channel: np.ndarray,
    alpha: float,
    strength: float,
    preserve_low_freq: float,
    smoothing: float,
    r_norm: np.ndarray,
    n_bins: int = 128,
) -> np.ndarray:
    """
    Apply radial spectrum normalisation to a single [H, W] channel.

    Parameters
    ----------
    channel           : [H, W] float32
    alpha             : exponent of target 1/f^alpha power law (typically ~2)
    strength          : blend factor 0=no change, 1=full correction
    preserve_low_freq : normalised radius below which gain is frozen at 1.0
    smoothing         : sigma for Gaussian smoothing of the radial gain curve
    r_norm            : [H, W] normalised radial distance grid (precomputed)
    n_bins            : number of radial bins for spectrum estimation
    """
    spectrum = fft2_channel(channel)
    mag      = magnitude(spectrum)
    ph       = phase(spectrum)

    power          = mag ** 2
    bin_centers, bin_means = _radial_average(power, r_norm, n_bins)
    target         = _build_target_curve(bin_centers, bin_means, alpha)

    # Amplitude gain = sqrt(target_power / current_power)
    safe_means = np.where(bin_means < 1e-12, 1e-12, bin_means)
    gain_bins  = np.sqrt(target / safe_means)

    # Smooth the gain curve to avoid abrupt radial transitions
    if smoothing > 0:
        sigma      = smoothing * n_bins   # smoothing in bin units
        gain_bins  = gaussian_filter1d(gain_bins, sigma=sigma, mode="nearest")

    # Freeze gain = 1 in the low-frequency band
    low_mask           = bin_centers < preserve_low_freq
    gain_bins[low_mask] = 1.0

    # Also freeze DC bin always
    gain_bins[0] = 1.0

    # Clip gain so we never boost or over-suppress by extreme factors
    gain_bins = np.clip(gain_bins, 0.01, 10.0)

    # Blend toward identity: G_final = 1 + strength * (G - 1)
    gain_bins = 1.0 + strength * (gain_bins - 1.0)

    # Interpolate onto 2-D grid
    gain_map = _radial_gain_map(r_norm, bin_centers, gain_bins)

    new_mag  = mag * gain_map
    new_spec = reconstruct(new_mag, ph)
    return ifft2_channel(new_spec)


# ---------------------------------------------------------------------------
# Per-image wrapper
# ---------------------------------------------------------------------------

def _normalize_radial_image(
    image: np.ndarray,
    alpha: float,
    strength: float,
    preserve_low_freq: float,
    smoothing: float,
) -> np.ndarray:
    h, w, c = image.shape
    r_norm  = radial_grid(h, w)

    out = np.empty_like(image)
    for ch in range(c):
        out[:, :, ch] = _normalize_radial_channel(
            image[:, :, ch],
            alpha=alpha,
            strength=strength,
            preserve_low_freq=preserve_low_freq,
            smoothing=smoothing,
            r_norm=r_norm,
        )
    return out


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class RadialSpectrumNormalizer:
    """
    Nudges the radial power spectrum of an image toward the natural 1/f^alpha
    statistical law that characterises real photographs.

    AI-generated or heavily processed images often carry excess high-frequency
    energy that can destabilise the VAE encoder.  This node applies a smooth
    radial gain correction — never a hard cutoff — to restore a more natural
    spectral shape.
    """

    CATEGORY = "Spectral Preprocessing"

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.5,
                        "max": 4.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": (
                            "Exponent of the target 1/f^alpha power law. "
                            "Natural photos ≈ 2.0.  Higher values push more energy "
                            "toward low frequencies."
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Blend factor between the original spectrum (0) and "
                            "the fully corrected spectrum (1).  Start low."
                        ),
                    },
                ),
                "preserve_low_freq": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Normalised radius (0–1) below which gain is frozen "
                            "at 1.0.  Protects global structure and brightness."
                        ),
                    },
                ),
                "smoothing": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 0.3,
                        "step": 0.005,
                        "display": "slider",
                        "tooltip": (
                            "Gaussian smoothing sigma applied to the radial gain "
                            "curve (as a fraction of the number of bins).  "
                            "Higher = smoother transition, less ringing."
                        ),
                    },
                ),
                "tile_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Tile size for large images. 0 = process whole image.",
                    },
                ),
                "tile_overlap": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 512,
                        "step": 32,
                        "tooltip": "Tile overlap in pixels (used only when tile_size > 0).",
                    },
                ),
            }
        }

    def apply(
        self,
        image: "torch.Tensor",
        alpha: float,
        strength: float,
        preserve_low_freq: float,
        smoothing: float,
        tile_size: int,
        tile_overlap: int,
    ) -> tuple:
        arr            = tensor_to_numpy(image)
        orig_h, orig_w = arr.shape[1], arr.shape[2]
        processed      = process_batch_tiled(
            arr,
            _normalize_radial_image,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            alpha=alpha,
            strength=strength,
            preserve_low_freq=preserve_low_freq,
            smoothing=smoothing,
        )
        processed = center_crop_to_match(processed, orig_h, orig_w)
        return (numpy_to_tensor(clamp01(processed)),)
