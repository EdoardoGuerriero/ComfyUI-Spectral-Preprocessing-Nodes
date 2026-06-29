"""
Node 8 — Spectral Channel Equalizer
======================================
Reduces inter-channel spectral mismatch to suppress color fringing artifacts.

AI upscaling and some generative models produce images where the R, G, and B
channels have different radial power spectra.  In a natural photograph the
three channels are close to spectrally balanced (same approximate power at
each frequency band).  When channels diverge, it manifests as:

  • Color fringing on edges
  • Chromatic aberration-like halos
  • Color noise that survives denoising
  • Saturation artifacts at high-frequency detail

Algorithm:
  1.  FFT each channel → magnitude R, G, B
  2.  Compute the radial power spectrum per channel  P_R(r), P_G(r), P_B(r)
  3.  Compute the reference spectrum  P_ref(r) = mean(P_R, P_G, P_B)
  4.  Build a per-channel radial gain:
        G_ch(r) = sqrt( P_ref(r) / P_ch(r) )
      (amplitude gain to match power to the reference)
  5.  Smooth the gain curves radially (no sharp frequency edges)
  6.  Optionally protect the low-frequency band (global colour balance)
  7.  Blend:  G_final = 1 + strength * (G − 1)
  8.  Apply per-channel gain map, reconstruct, IFFT

The reference spectrum is the *average* of the three channels, so no channel
is boosted or cut in absolute terms — only the relative spectral balance
between channels is corrected.  Global colour (DC and near-DC) is preserved
by the `preserve_low_freq` guard.
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
# Radial power profile helpers
# ---------------------------------------------------------------------------

def _radial_power(mag: np.ndarray, r_norm: np.ndarray, n_bins: int) -> tuple:
    """
    Return (bin_centers [n_bins], mean_power [n_bins]) for a magnitude array.
    """
    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    power   = mag ** 2
    means   = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        mask = (r_norm >= edges[i]) & (r_norm < edges[i + 1])
        if mask.any():
            means[i] = power[mask].mean()

    # Fill any empty bins with neighbour interpolation
    nans = means == 0
    if nans.any() and not nans.all():
        idx = np.arange(n_bins)
        means[nans] = np.interp(idx[nans], idx[~nans], means[~nans])

    return centers, means


def _build_channel_gain(
    power_ch: np.ndarray,
    power_ref: np.ndarray,
    smoothing_sigma: float,
    preserve_low_freq: float,
    strength: float,
    bin_centers: np.ndarray,
) -> np.ndarray:
    """
    Build the radial gain curve for one channel.

    gain = sqrt(P_ref / P_ch)  →  amplitude gain that equalises power to ref.
    """
    safe_ch  = np.where(power_ch < 1e-12, 1e-12, power_ch)
    gain     = np.sqrt(power_ref / safe_ch)

    # Smooth to prevent abrupt per-bin transitions
    if smoothing_sigma > 0:
        gain = gaussian_filter1d(gain, sigma=smoothing_sigma, mode="nearest")

    # Clip to prevent extreme corrections
    gain = np.clip(gain, 0.05, 20.0)

    # Freeze low frequencies (preserve global colour balance)
    gain[bin_centers < preserve_low_freq] = 1.0
    gain[0] = 1.0  # always protect DC

    # Blend toward identity
    gain = 1.0 + strength * (gain - 1.0)

    return gain


# ---------------------------------------------------------------------------
# Core algorithm (single image — operates across all 3 channels together)
# ---------------------------------------------------------------------------

def _equalize_channels_image(
    image: np.ndarray,
    strength: float,
    preserve_low_freq: float,
    smoothing: float,
    n_bins: int,
) -> np.ndarray:
    h, w, c = image.shape
    r_norm  = radial_grid(h, w)

    # --- compute spectra for all channels ---
    spectra = []
    mags    = []
    phases  = []
    for ch in range(c):
        s = fft2_channel(image[:, :, ch])
        spectra.append(s)
        mags.append(magnitude(s))
        phases.append(phase(s))

    # --- radial power per channel ---
    bin_centers_list = []
    powers           = []
    for ch in range(c):
        bc, pw = _radial_power(mags[ch], r_norm, n_bins)
        bin_centers_list.append(bc)
        powers.append(pw)

    bin_centers = bin_centers_list[0]   # same grid for all channels

    # --- reference = mean power across channels ---
    power_ref   = np.mean(np.stack(powers, axis=0), axis=0)

    # --- smoothing sigma in bin units ---
    sigma_bins  = smoothing * n_bins

    # --- build gain maps and apply ---
    out = np.empty_like(image)
    for ch in range(c):
        gain_bins = _build_channel_gain(
            power_ch=powers[ch],
            power_ref=power_ref,
            smoothing_sigma=sigma_bins,
            preserve_low_freq=preserve_low_freq,
            strength=strength,
            bin_centers=bin_centers,
        )
        # Interpolate 1-D gain curve onto 2-D radial grid
        gain_map = np.interp(r_norm.ravel(), bin_centers, gain_bins) \
                     .reshape(r_norm.shape).astype(np.float32)

        new_mag      = mags[ch] * gain_map
        new_spec     = reconstruct(new_mag, phases[ch])
        out[:, :, ch] = ifft2_channel(new_spec)

    return out


# ---------------------------------------------------------------------------
# Batch wrapper
# ---------------------------------------------------------------------------

def _equalize_channels_batch_fn(
    image: np.ndarray,
    strength: float,
    preserve_low_freq: float,
    smoothing: float,
    n_bins: int,
) -> np.ndarray:
    return _equalize_channels_image(image, strength, preserve_low_freq, smoothing, n_bins)


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class SpectralChannelEqualizer:
    """
    Reduces spectral imbalance between the R, G, and B channels.

    Each channel's radial power spectrum is measured, compared against the
    mean of all three channels, and a smooth radial gain is applied to nudge
    all channels toward a common spectral shape.

    Targets: colour fringing, chromatic halos, colour noise, and saturation
    artifacts at high-frequency detail — common in AI-upscaled and
    AI-generated images.

    Global colour balance (low frequencies) is protected by the
    preserve_low_freq parameter.
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
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "How strongly to equalise the channel spectra. "
                            "0 = no change, 1 = full equalization toward the mean spectrum."
                        ),
                    },
                ),
                "preserve_low_freq": (
                    "FLOAT",
                    {
                        "default": 0.08,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Normalised radius below which channel gains are frozen at 1.0.  "
                            "Protects global colour balance and image tone."
                        ),
                    },
                ),
                "smoothing": (
                    "FLOAT",
                    {
                        "default": 0.06,
                        "min": 0.0,
                        "max": 0.3,
                        "step": 0.005,
                        "display": "slider",
                        "tooltip": (
                            "Gaussian smoothing sigma for the radial gain curves "
                            "(fraction of n_bins).  Higher = smoother correction."
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
                        "tooltip": "Tile size for large images. 0 = whole image.",
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
            _equalize_channels_batch_fn,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            strength=strength,
            preserve_low_freq=preserve_low_freq,
            smoothing=smoothing,
            n_bins=128,
        )
        processed = center_crop_to_match(processed, orig_h, orig_w)
        return (numpy_to_tensor(clamp01(processed)),)
