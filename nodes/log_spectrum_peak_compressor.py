"""
Node 5 — Log Spectrum Peak Compressor
=======================================
Processes the FFT magnitude in the logarithmic domain, inspired by cepstral
and homomorphic signal processing techniques from audio engineering.

Working in log space turns multiplicative spectral distortions (gain patterns,
grid artifacts, repetitive texture overlays) into additive signals that are
much easier to separate from the underlying image content.

Algorithm:
  1.  FFT → magnitude
  2.  log_mag = log(magnitude + ε)           (shift to log domain)
  3.  background = smooth(log_mag)            (Gaussian or uniform blur)
  4.  residual = log_mag − background         (local peaks in log space)
  5.  compressed_residual = soft_compress(residual, peak_threshold, ratio)
  6.  log_mag_out = background + compressed_residual
  7.  magnitude_out = exp(log_mag_out)        (back to linear)
  8.  Reconstruct complex spectrum from original phase
  9.  Inverse FFT

The background estimation is performed with a 2-D spatial blur of the
log-magnitude — this captures the smooth spectral envelope and lets the
residual contain only the fine structure (peaks, spikes, narrow bands).

Because we work in log space:
  • A spike that is 10× the background in linear becomes +log(10) ≈ 2.3 above
    the background in log space — easy to detect and compress.
  • The phase is never touched, so spatial structure is preserved.
  • Compression in log space = ratio reduction in linear space.

The soft-clip compressor used on the residual:
    out = sign(r) * threshold * log(1 + |r| / threshold) * ratio_factor

where ratio_factor shapes how aggressively peaks above `peak_threshold`
are reduced relative to their height.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from ..utils.fft_utils import (
    tensor_to_numpy,
    numpy_to_tensor,
    fft2_channel,
    ifft2_channel,
    magnitude,
    phase,
    reconstruct,
    get_dc_mask,
    process_batch_tiled,
    center_crop_to_match,
    clamp01,
)


# ---------------------------------------------------------------------------
# Background estimation in log-magnitude space
# ---------------------------------------------------------------------------

def _estimate_log_background(log_mag: np.ndarray, smoothing: float, mode: str) -> np.ndarray:
    """
    Estimate the smooth spectral envelope in log-magnitude space.

    smoothing : sigma (Gaussian) or kernel half-size (uniform), as fraction of
                the smaller image dimension
    mode      : 'gaussian' or 'uniform'
    """
    h, w   = log_mag.shape
    sigma  = smoothing * min(h, w)

    if mode == "gaussian":
        return gaussian_filter(log_mag, sigma=sigma, mode="reflect")
    else:
        size = max(3, int(sigma * 2 + 1))
        if size % 2 == 0:
            size += 1
        return uniform_filter(log_mag, size=size, mode="reflect")


# ---------------------------------------------------------------------------
# Soft-log compressor applied to residual
# ---------------------------------------------------------------------------

def _compress_residual(
    residual: np.ndarray,
    peak_threshold: float,
    ratio: float,
) -> np.ndarray:
    """
    Apply a soft-log compressor to the log-domain residual.

    Values with |residual| < peak_threshold are passed through unchanged.
    Values above are compressed: the excess is reduced by the given ratio.

    This is equivalent to a soft-knee compressor in log-log space, which
    translates to a very gentle power-law compression in linear space.
    """
    abs_r = np.abs(residual)
    sign  = np.sign(residual)

    # Below threshold: identity
    # Above threshold: threshold + (excess) / ratio
    excess      = np.maximum(0.0, abs_r - peak_threshold)
    compressed  = abs_r - excess + excess / ratio

    return sign * compressed


# ---------------------------------------------------------------------------
# Core algorithm (single channel)
# ---------------------------------------------------------------------------

def _log_compress_channel(
    channel: np.ndarray,
    peak_threshold: float,
    ratio: float,
    smoothing: float,
    strength: float,
    dc_mask: np.ndarray,
    bg_mode: str,
    epsilon: float,
) -> np.ndarray:
    """
    Apply log-spectrum peak compression to a single [H, W] channel.

    Parameters
    ----------
    channel        : [H, W] float32
    peak_threshold : threshold in log-magnitude units for the residual compressor
    ratio          : compression ratio for the residual
    smoothing      : background blur strength (fraction of image size)
    strength       : blend factor (0 = no change, 1 = full compression)
    dc_mask        : [H, W] bool — these pixels are reconstructed from original mag
    bg_mode        : background estimator, 'gaussian' or 'uniform'
    epsilon        : small constant added before log to avoid log(0)
    """
    spectrum = fft2_channel(channel)
    mag      = magnitude(spectrum)
    ph       = phase(spectrum)

    # --- log domain -------------------------------------------------------
    log_mag    = np.log(mag + epsilon)
    background = _estimate_log_background(log_mag, smoothing, bg_mode)
    residual   = log_mag - background

    # --- compress residual ------------------------------------------------
    compressed_residual = _compress_residual(residual, peak_threshold, ratio)

    # --- reconstruct log magnitude ----------------------------------------
    log_mag_out = background + compressed_residual
    new_mag     = np.exp(log_mag_out) - epsilon
    new_mag     = np.maximum(new_mag, 0.0)

    # --- blend with original ----------------------------------------------
    blended_mag = mag + strength * (new_mag - mag)

    # Restore original magnitude at DC
    blended_mag[dc_mask] = mag[dc_mask]

    new_spec = reconstruct(blended_mag, ph)
    return ifft2_channel(new_spec)


# ---------------------------------------------------------------------------
# Per-image wrapper
# ---------------------------------------------------------------------------

def _log_compress_image(
    image: np.ndarray,
    peak_threshold: float,
    ratio: float,
    smoothing: float,
    strength: float,
    bg_mode: str,
    epsilon: float,
) -> np.ndarray:
    h, w, c = image.shape
    dc_mask = get_dc_mask(h, w, radius=2)

    out = np.empty_like(image)
    for ch in range(c):
        out[:, :, ch] = _log_compress_channel(
            image[:, :, ch],
            peak_threshold=peak_threshold,
            ratio=ratio,
            smoothing=smoothing,
            strength=strength,
            dc_mask=dc_mask,
            bg_mode=bg_mode,
            epsilon=epsilon,
        )
    return out


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class LogSpectrumPeakCompressor:
    """
    Compresses peaks in the FFT log-magnitude spectrum relative to the smooth
    spectral background.

    Inspired by homomorphic filtering and cepstral processing from audio
    engineering: working in log space separates the spectral envelope (image
    content) from fine spectral structure (artifacts), making the peaks easy
    to compress without harming the underlying signal.

    Unlike hard thresholding, the soft-log compressor gracefully reduces
    peaks above the threshold rather than clipping them, producing minimal
    ringing in the spatial domain.
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
                "peak_threshold": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": (
                            "Threshold for the log-domain residual.  "
                            "Values above this are compressed.  Smaller = more aggressive.  "
                            "Typical artifact peaks sit at 1–3 log units above background."
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
                            "Compression ratio applied to the log-domain excess above "
                            "the threshold.  4.0 = a 4-unit excess becomes 1 unit."
                        ),
                    },
                ),
                "smoothing": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.005,
                        "max": 0.3,
                        "step": 0.005,
                        "display": "slider",
                        "tooltip": (
                            "Background blur strength as a fraction of image size.  "
                            "Larger values = broader envelope estimate = only very narrow "
                            "spikes are treated as 'peaks'."
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Blend between original (0) and fully compressed (1).",
                    },
                ),
                "background_mode": (
                    ["gaussian", "uniform"],
                    {
                        "default": "gaussian",
                        "tooltip": (
                            "Background estimation method.  Gaussian gives a smoother "
                            "envelope; uniform is faster and sharper."
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
        peak_threshold: float,
        ratio: float,
        smoothing: float,
        strength: float,
        background_mode: str,
        tile_size: int,
        tile_overlap: int,
    ) -> tuple:
        epsilon        = 1e-6
        arr            = tensor_to_numpy(image)
        orig_h, orig_w = arr.shape[1], arr.shape[2]
        processed      = process_batch_tiled(
            arr,
            _log_compress_image,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            peak_threshold=peak_threshold,
            ratio=ratio,
            smoothing=smoothing,
            strength=strength,
            bg_mode=background_mode,
            epsilon=epsilon,
        )
        processed = center_crop_to_match(processed, orig_h, orig_w)
        return (numpy_to_tensor(clamp01(processed)),)
