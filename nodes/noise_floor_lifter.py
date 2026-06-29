"""
Node 7 — Noise Floor Lifter
=============================
Estimates and subtracts the spectral noise floor from the FFT magnitude.

AI-generated images, compressed images, and upscaled images often exhibit
an elevated, spatially-uniform noise floor across the frequency spectrum —
energy that exists at roughly the same level at all frequencies and angles.
In natural photographs this background level is much lower relative to the
structured content.

In audio this operation is called *spectral subtraction*: estimate the noise
floor, subtract it, clip negatives to zero.  The result is a spectrum where
genuine signal peaks stand out more clearly against a cleaner background.

Algorithm:
  1.  FFT → magnitude
  2.  Estimate noise floor = percentile(magnitude, floor_percentile)
      Using a low percentile (e.g. 10th) gives a robust estimate of the
      "quietest" level — genuine signal peaks pull the mean up but barely
      affect a low percentile.
  3.  Subtract floor * strength from every magnitude coefficient
  4.  Clip negatives to zero (Wiener-like: never go below 0)
  5.  Optionally apply a spectral over-subtraction factor to reduce
      residual musical noise (analogous to the β parameter in audio)
  6.  Reconstruct + IFFT

Because the subtraction is global and uniform (same amount subtracted
everywhere), it does NOT introduce any spatial ringing — there are no
spatial-frequency edges in the gain mask.

Parameters
----------
floor_percentile : percentile of the magnitude distribution used as noise estimate
strength         : fraction of the estimated floor to subtract (0 = none, 1 = full)
over_subtraction : multiplier on the floor estimate before subtraction
                   Values > 1 subtract more aggressively but may notch weak signal.
preserve_dc      : always leave the DC component unmodified
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
    get_dc_mask,
    process_batch_tiled,
    center_crop_to_match,
    clamp01,
)


# ---------------------------------------------------------------------------
# Core algorithm (single channel)
# ---------------------------------------------------------------------------

def _lift_noise_floor_channel(
    channel: np.ndarray,
    floor_percentile: float,
    strength: float,
    over_subtraction: float,
    dc_mask: np.ndarray,
) -> np.ndarray:
    spectrum  = fft2_channel(channel)
    mag       = magnitude(spectrum)
    ph        = phase(spectrum)

    # Estimate noise floor from the lowest-energy coefficients
    floor_level = np.percentile(mag, floor_percentile)

    # Spectral subtraction with over-subtraction factor
    subtract = strength * over_subtraction * floor_level
    new_mag  = np.maximum(mag - subtract, 0.0)

    # DC protection
    new_mag[dc_mask] = mag[dc_mask]

    new_spec = reconstruct(new_mag, ph)
    return ifft2_channel(new_spec)


# ---------------------------------------------------------------------------
# Per-image wrapper
# ---------------------------------------------------------------------------

def _lift_noise_floor_image(
    image: np.ndarray,
    floor_percentile: float,
    strength: float,
    over_subtraction: float,
) -> np.ndarray:
    h, w, c = image.shape
    dc_mask = get_dc_mask(h, w, radius=2)

    out = np.empty_like(image)
    for ch in range(c):
        out[:, :, ch] = _lift_noise_floor_channel(
            image[:, :, ch],
            floor_percentile=floor_percentile,
            strength=strength,
            over_subtraction=over_subtraction,
            dc_mask=dc_mask,
        )
    return out


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class NoiseFloorLifter:
    """
    Estimates the spectral noise floor (the uniform background energy level
    in the FFT magnitude) and subtracts it.

    Analogous to spectral subtraction in audio processing.  Particularly
    effective on AI-generated images and heavily compressed sources where
    the noise floor is elevated relative to natural photographs.

    Unlike spatial denoising, this operates entirely in frequency space and
    does not blur edges or reduce fine detail — it only removes energy that
    was uniformly distributed across the spectrum.
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
                "floor_percentile": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 1.0,
                        "max": 49.0,
                        "step": 0.5,
                        "display": "slider",
                        "tooltip": (
                            "Percentile of the magnitude distribution used as the "
                            "noise floor estimate.  10 = the 10th percentile of all "
                            "FFT coefficients.  Lower = more conservative estimate."
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Fraction of the noise floor estimate to subtract.",
                    },
                ),
                "over_subtraction": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 1.0,
                        "max": 3.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": (
                            "Multiplier on the floor estimate.  Values > 1 subtract more "
                            "aggressively, useful for strongly contaminated images, "
                            "but may remove weak genuine signal at high values."
                        ),
                    },
                ),
                "preserve_dc": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Leave the DC coefficient (mean brightness) untouched.",
                    },
                ),
                "tile_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2048,
                        "step": 64,
                        "tooltip": (
                            "Tile size for large images.  0 = process whole image at once.  "
                            "Use 512–1024 for images larger than 2048px."
                        ),
                    },
                ),
                "tile_overlap": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 512,
                        "step": 32,
                        "tooltip": "Overlap in pixels between adjacent tiles (used only when tile_size > 0).",
                    },
                ),
            }
        }

    def apply(
        self,
        image: "torch.Tensor",
        floor_percentile: float,
        strength: float,
        over_subtraction: float,
        preserve_dc: bool,
        tile_size: int,
        tile_overlap: int,
    ) -> tuple:
        arr            = tensor_to_numpy(image)
        orig_h, orig_w = arr.shape[1], arr.shape[2]
        processed      = process_batch_tiled(
            arr,
            _lift_noise_floor_image,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            floor_percentile=floor_percentile,
            strength=strength,
            over_subtraction=over_subtraction,
        )
        processed = center_crop_to_match(processed, orig_h, orig_w)
        return (numpy_to_tensor(clamp01(processed)),)
