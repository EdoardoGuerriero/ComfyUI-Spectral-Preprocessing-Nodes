"""
Node 1 — Spectral Spike Suppressor
===================================
Detects and attenuates isolated peaks in the FFT magnitude spectrum without
touching the surrounding frequencies.  This targets periodic artifacts that
appear as narrow spikes in frequency space:

  • JPEG mosquito noise / ringing
  • Checkerboard patterns (from transposed convolutions)
  • Moiré fringes
  • CNN / upscaling grid artifacts
  • AI-generated image compression artifacts

Algorithm (per channel):
  1.  FFT → shifted magnitude + phase
  2.  Estimate smooth background via median filter
  3.  Compute residual  =  magnitude − background
  4.  Normalise residual by local standard deviation → z-score map
  5.  Build a soft attenuation mask:
        gain = 1  where z < threshold_sigma   (untouched)
        gain smoothly falls to (1 − attenuation)  where z >> threshold_sigma
  6.  Optionally protect the DC component from any gain change
  7.  Apply gain to magnitude, reconstruct complex spectrum from original phase
  8.  Inverse FFT → clamp → output

The phase is always preserved exactly; only the magnitude is modified.
This guarantees that spatial structure (edges, textures) is not distorted
beyond the amplitude change — ringing is therefore minimal.
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
    estimate_background_median,
    get_dc_mask,
    stable_sigmoid,
    process_batch_tiled,
    center_crop_to_match,
    clamp01,
)


# ---------------------------------------------------------------------------
# Core algorithm (single [H, W] channel)
# ---------------------------------------------------------------------------

def _suppress_spikes_channel(
    channel: np.ndarray,
    threshold_sigma: float,
    attenuation: float,
    kernel_size: int,
    dc_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply spike suppression to a single float32 [H, W] channel.

    Parameters
    ----------
    channel         : [H, W] float32, values in [0, 1]
    threshold_sigma : z-score above which a coefficient is treated as a spike
    attenuation     : fraction of spike energy to remove (0 = nothing, 1 = full)
    kernel_size     : median filter kernel size for background estimation
    dc_mask         : boolean [H, W] mask; True pixels are never attenuated

    Returns
    -------
    [H, W] float32, same range as input
    """
    spectrum = fft2_channel(channel)
    mag = magnitude(spectrum)
    ph  = phase(spectrum)

    # --- background estimation -------------------------------------------
    background = estimate_background_median(mag, kernel_size)

    # --- residual z-score ------------------------------------------------
    residual = mag - background
    # local std estimated from the residual itself (robust: use MAD → σ)
    mad = np.median(np.abs(residual - np.median(residual)))
    local_std = mad * 1.4826 + 1e-8   # MAD → Gaussian σ equivalent

    z = residual / local_std           # z-score map

    # --- soft attenuation mask -------------------------------------------
    # Sigmoid-like: gain drops from 1 toward (1 - attenuation) as z grows
    # past threshold_sigma.  Using a smooth logistic so there's no hard edge.
    overshoot = z - threshold_sigma    # negative → below threshold (gain=1)
    sigmoid   = stable_sigmoid(overshoot * 2.0)
    gain      = 1.0 - attenuation * sigmoid

    # clips: gain must stay in [1-attenuation, 1], never boost
    gain = np.clip(gain, 1.0 - attenuation, 1.0)

    # --- DC protection ---------------------------------------------------
    gain[dc_mask] = 1.0

    # --- reconstruct -----------------------------------------------------
    new_mag  = mag * gain
    new_spec = reconstruct(new_mag, ph)

    return ifft2_channel(new_spec)


# ---------------------------------------------------------------------------
# Per-image wrapper (processes all RGB channels)
# ---------------------------------------------------------------------------

def _suppress_spikes_image(
    image: np.ndarray,
    threshold_sigma: float,
    attenuation: float,
    kernel_size: int,
    preserve_dc: bool,
) -> np.ndarray:
    """
    Apply spike suppression to a single [H, W, C] image.
    Each channel is processed independently.
    """
    h, w, c = image.shape
    dc_mask = get_dc_mask(h, w) if preserve_dc else np.zeros((h, w), dtype=bool)

    out = np.empty_like(image)
    for ch in range(c):
        out[:, :, ch] = _suppress_spikes_channel(
            image[:, :, ch],
            threshold_sigma=threshold_sigma,
            attenuation=attenuation,
            kernel_size=kernel_size,
            dc_mask=dc_mask,
        )
    return out


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class SpectralSpikeSuppressor:
    """
    Attenuates isolated spikes in the FFT magnitude spectrum while preserving
    all other frequency content and the complete phase information.

    Useful as a first preprocessing step before Flux image-to-image to remove
    periodic artifacts that destabilise the VAE encoder.
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
                "threshold_sigma": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.5,
                        "max": 20.0,
                        "step": 0.1,
                        "display": "slider",
                        "tooltip": (
                            "Z-score threshold above which a frequency coefficient "
                            "is considered a spike.  Lower values are more aggressive."
                        ),
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
                        "tooltip": (
                            "Maximum fraction of spike energy to remove. "
                            "1.0 = full suppression, 0.0 = no change."
                        ),
                    },
                ),
                "kernel_size": (
                    "INT",
                    {
                        "default": 15,
                        "min": 3,
                        "max": 63,
                        "step": 2,
                        "tooltip": (
                            "Median filter kernel size for background estimation. "
                            "Must be odd; larger values capture broader structure."
                        ),
                    },
                ),
                "preserve_dc": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Protect the DC component (mean brightness) from any "
                            "attenuation.  Recommended: True."
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
        threshold_sigma: float,
        attenuation: float,
        kernel_size: int,
        preserve_dc: bool,
        tile_size: int,
        tile_overlap: int,
    ) -> tuple:
        if kernel_size % 2 == 0:
            kernel_size += 1

        arr            = tensor_to_numpy(image)
        orig_h, orig_w = arr.shape[1], arr.shape[2]
        processed      = process_batch_tiled(
            arr,
            _suppress_spikes_image,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            threshold_sigma=threshold_sigma,
            attenuation=attenuation,
            kernel_size=kernel_size,
            preserve_dc=preserve_dc,
        )
        processed = center_crop_to_match(processed, orig_h, orig_w)
        return (numpy_to_tensor(clamp01(processed)),)
