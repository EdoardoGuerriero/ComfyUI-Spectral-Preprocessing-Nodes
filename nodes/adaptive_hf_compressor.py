"""
Node 3 — Adaptive High-Frequency Compressor
=============================================
Applies dynamics compression (in the audio sense) to the high-frequency band
of the FFT magnitude spectrum.

The key insight borrowed from audio engineering:
  • A limiter hard-clips peaks → distortion (spatial ringing here)
  • A compressor softly reduces peaks above a threshold → transparent

When the high-frequency energy is below `threshold`, nothing changes.
When it exceeds `threshold`, only the excess is reduced, following:

    output_HF = threshold + (input_HF - threshold) / ratio

A soft knee blends smoothly between the linear and compressed regions,
preventing an abrupt gain change at exactly the threshold.

This is applied as a *gain envelope* across the entire high-frequency region
so that the relative shape of the spectrum is preserved — only the overall
level is reduced.

Algorithm:
  1.  FFT → magnitude + phase
  2.  Build a radial mask separating low-freq (≤ hf_cutoff) from high-freq
  3.  Compute RMS energy of the high-frequency band
  4.  Apply soft-knee compression to determine the output RMS
  5.  Derive a scalar gain  G = output_RMS / input_RMS
  6.  Blend: effective_gain = 1 + strength * (G - 1)
  7.  Apply effective_gain to high-frequency region only
  8.  Reconstruct + IFFT
"""

import numpy as np

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
# Soft-knee compression
# ---------------------------------------------------------------------------

def _soft_knee_gain(rms: float, threshold: float, ratio: float, knee: float) -> float:
    """
    Compute the amplitude gain scalar for a soft-knee compressor.

    Parameters
    ----------
    rms       : current RMS level (linear amplitude)
    threshold : level above which compression starts (linear amplitude)
    ratio     : compression ratio  (e.g. 4.0 = 4:1)
    knee      : width of the soft-knee transition in the same units as rms

    Returns
    -------
    gain scalar ∈ (0, 1]  — never boosts
    """
    if ratio <= 1.0:
        return 1.0   # No compression

    # Work in log domain for the knee calculation, then convert back
    # to avoid issues when rms or threshold are near zero
    if rms < 1e-10:
        return 1.0

    diff = rms - threshold

    if knee > 0:
        # Smooth transition region [-knee/2, +knee/2] around threshold
        half_knee = knee / 2.0
        if diff < -half_knee:
            # Below knee: no gain change
            return 1.0
        elif diff > half_knee:
            # Above knee: full compression
            compressed = threshold + diff / ratio
            return compressed / rms
        else:
            # Inside knee: blend linearly
            t = (diff + half_knee) / knee   # 0 → 1 across the knee
            compressed_full = threshold + diff / ratio
            compressed_none = rms                       # identity
            output = (1 - t) * compressed_none + t * compressed_full
            return output / rms
    else:
        # Hard knee
        if diff <= 0:
            return 1.0
        compressed = threshold + diff / ratio
        return compressed / rms


# ---------------------------------------------------------------------------
# Core algorithm (single channel)
# ---------------------------------------------------------------------------

def _compress_hf_channel(
    channel: np.ndarray,
    threshold: float,
    ratio: float,
    knee: float,
    strength: float,
    hf_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply adaptive HF compression to a single [H, W] channel.

    Parameters
    ----------
    channel   : [H, W] float32
    threshold : RMS threshold in magnitude units above which compression kicks in
    ratio     : compression ratio
    knee      : soft-knee width in the same units as threshold
    strength  : blend factor (0 = no change, 1 = full compression)
    hf_mask   : boolean [H, W], True = high-frequency region
    """
    spectrum = fft2_channel(channel)
    mag      = magnitude(spectrum)
    ph       = phase(spectrum)

    # RMS of the HF band
    hf_mag = mag[hf_mask]
    if hf_mag.size == 0:
        return channel

    rms_in = float(np.sqrt(np.mean(hf_mag ** 2)))

    # Scalar compression gain for the whole HF band
    g_compress = _soft_knee_gain(rms_in, threshold, ratio, knee)

    # Blend toward identity
    g_effective = 1.0 + strength * (g_compress - 1.0)
    g_effective = float(np.clip(g_effective, 0.0, 1.0))

    new_mag         = mag.copy()
    new_mag[hf_mask] = hf_mag * g_effective

    new_spec = reconstruct(new_mag, ph)
    return ifft2_channel(new_spec)


# ---------------------------------------------------------------------------
# Per-image wrapper
# ---------------------------------------------------------------------------

def _compress_hf_image(
    image: np.ndarray,
    threshold: float,
    ratio: float,
    knee: float,
    strength: float,
    hf_cutoff: float,
) -> np.ndarray:
    h, w, c = image.shape
    r_norm  = radial_grid(h, w)
    hf_mask = r_norm >= hf_cutoff

    out = np.empty_like(image)
    for ch in range(c):
        out[:, :, ch] = _compress_hf_channel(
            image[:, :, ch],
            threshold=threshold,
            ratio=ratio,
            knee=knee,
            strength=strength,
            hf_mask=hf_mask,
        )
    return out


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class AdaptiveHFCompressor:
    """
    Applies soft-knee dynamics compression to the high-frequency band of the
    FFT magnitude spectrum.

    Like an audio compressor: if the HF energy is below the threshold,
    nothing changes.  Only excess energy above the threshold is reduced,
    following a configurable ratio and soft knee.  The relative spectral
    shape within the HF band is preserved — only its overall level is adjusted.
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
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.001,
                        "max": 1.0,
                        "step": 0.005,
                        "display": "slider",
                        "tooltip": (
                            "RMS magnitude threshold.  HF energy below this level "
                            "is untouched.  Typical FFT magnitudes for natural images "
                            "are in the 0.05–0.3 range at high frequencies."
                        ),
                    },
                ),
                "ratio": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.5,
                        "display": "slider",
                        "tooltip": (
                            "Compression ratio.  4:1 means that 4 dB above the "
                            "threshold becomes 1 dB above it.  Higher = more aggressive."
                        ),
                    },
                ),
                "knee": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 0.3,
                        "step": 0.005,
                        "display": "slider",
                        "tooltip": (
                            "Soft-knee width.  0 = hard knee (abrupt).  "
                            "Larger values create a smoother transition around "
                            "the threshold, reducing spatial ringing."
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Blend between original (0) and fully compressed (1).",
                    },
                ),
                "hf_cutoff": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.05,
                        "max": 0.9,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Normalised radial frequency (0–1) above which the signal "
                            "is considered 'high frequency'.  0.25 = top 75%% of the "
                            "frequency range."
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
        threshold: float,
        ratio: float,
        knee: float,
        strength: float,
        hf_cutoff: float,
        tile_size: int,
        tile_overlap: int,
    ) -> tuple:
        arr            = tensor_to_numpy(image)
        orig_h, orig_w = arr.shape[1], arr.shape[2]
        processed      = process_batch_tiled(
            arr,
            _compress_hf_image,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            threshold=threshold,
            ratio=ratio,
            knee=knee,
            strength=strength,
            hf_cutoff=hf_cutoff,
        )
        processed = center_crop_to_match(processed, orig_h, orig_w)
        return (numpy_to_tensor(clamp01(processed)),)
