"""
Node 11 — Spectral Histogram Match
=====================================
Matches the statistical distribution of FFT coefficient magnitudes from a
source image to those of a target image — the frequency-domain analogue of
histogram matching in pixel space.

Background
----------
Standard histogram matching remaps pixel values so the output's CDF matches
the target's CDF.  This node does exactly the same operation, but on the
*FFT magnitude spectrum* rather than on pixel values.

The key difference from pixel-space histogram matching:
  • Pixel matching operates on spatial amplitudes → changes colour and tone.
  • Spectral matching operates on frequency amplitudes → changes texture
    character, grain, sharpness, and spectral noise statistics.
    Colour and spatial structure (edges, object positions) are preserved
    because the phase is never modified.

What it achieves
----------------
After matching, the output image has the same *distribution of spectral
energy* as the target — how much energy is concentrated in rare large
coefficients vs spread across many small ones, how fat the magnitude tail is,
etc.  This is strictly more expressive than `RadialSpectrumNormalizer` (which
only matches the radial mean) — it matches the full statistical shape.

Typical uses
------------
Pre-processing (before VAE encode):
  • Match a synthetic/AI-generated source to a clean natural photograph to
    make its spectral statistics more natural.

Post-processing (after generation):
  • Match the generated output back to the source image to give it the same
    "film grain / camera texture" character — without touching colours or
    generated content, since phase is preserved.

Algorithm (per channel)
-----------------------
  1.  FFT(source) → mag_src, phase_src
  2.  FFT(target) → mag_tgt
  3.  Flatten both magnitude arrays
  4.  Sort source magnitudes → rank permutation
  5.  Sort target magnitudes → sorted target values
  6.  If source and target have different resolutions, resample the sorted
      target array to the same length as source via CDF interpolation
  7.  Assign: the source coefficient with rank k receives the target value
      at rank k
  8.  Reshape back to 2-D
  9.  Blend:  mag_out = (1 − strength) * mag_src + strength * mag_matched
  10. Protect DC: mag_out[DC] = mag_src[DC]  (preserves mean brightness)
  11. Reconstruct complex spectrum from mag_out and original phase_src
  12. IFFT

Note on different resolutions
------------------------------
Source and target may have different spatial resolutions.  The magnitude
arrays then have different numbers of coefficients.  Matching is performed
by resampling the sorted target values onto the source's CDF axis (linear
interpolation on the normalised CDF), which is the standard approach for
histograms with different bin counts.
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
    resize_to_match,
    get_dc_mask,
    process_batch_tiled,
    center_crop_to_match,
    clamp01,
)


# ---------------------------------------------------------------------------
# Core rank-matching helper
# ---------------------------------------------------------------------------

def _rank_match_magnitudes(mag_src: np.ndarray, mag_tgt: np.ndarray) -> np.ndarray:
    """
    Remap values of mag_src so their CDF matches mag_tgt.

    Works for arrays of any shape and handles different sizes via CDF
    interpolation on the normalised [0, 1] quantile axis.

    Returns an array with the same shape as mag_src.
    """
    src_flat = mag_src.ravel().astype(np.float64)
    tgt_flat = mag_tgt.ravel().astype(np.float64)

    # Rank permutation of source (stable sort preserves original order for ties)
    src_sort_idx = np.argsort(src_flat, kind="stable")

    # Sorted target values
    tgt_sorted = np.sort(tgt_flat)

    n_src = len(src_flat)
    n_tgt = len(tgt_flat)

    if n_src != n_tgt:
        # Resample sorted target onto the source quantile grid
        tgt_resampled = np.interp(
            np.linspace(0.0, 1.0, n_src),
            np.linspace(0.0, 1.0, n_tgt),
            tgt_sorted,
        )
    else:
        tgt_resampled = tgt_sorted

    # Assign: source rank k → target value at rank k
    new_flat = np.empty_like(src_flat)
    new_flat[src_sort_idx] = tgt_resampled

    return new_flat.reshape(mag_src.shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Core algorithm (single channel)
# ---------------------------------------------------------------------------

def _histogram_match_channel(
    ch_src: np.ndarray,
    ch_tgt: np.ndarray,
    strength: float,
    dc_mask: np.ndarray,
) -> np.ndarray:
    spec_src = fft2_channel(ch_src)
    spec_tgt = fft2_channel(ch_tgt)

    mag_src = magnitude(spec_src)
    mag_tgt = magnitude(spec_tgt)
    ph_src  = phase(spec_src)

    mag_matched = _rank_match_magnitudes(mag_src, mag_tgt)

    # Blend
    mag_out = (1.0 - strength) * mag_src + strength * mag_matched

    # DC protection — never change mean brightness
    mag_out[dc_mask] = mag_src[dc_mask]

    new_spec = reconstruct(mag_out.astype(np.float64), ph_src)
    return ifft2_channel(new_spec)


# ---------------------------------------------------------------------------
# Per-image wrapper
# ---------------------------------------------------------------------------

def _histogram_match_image(
    image_src: np.ndarray,
    image_tgt: np.ndarray,
    strength: float,
) -> np.ndarray:
    h, w, c = image_src.shape
    dc_mask = get_dc_mask(h, w, radius=2)

    # Resize target to match source resolution for the FFT (target is only
    # used for its magnitude statistics, not its spatial content)
    tgt_r = resize_to_match(image_tgt, h, w)

    out = np.empty_like(image_src)
    for ch in range(c):
        out[:, :, ch] = _histogram_match_channel(
            image_src[:, :, ch],
            tgt_r[:, :, ch],
            strength=strength,
            dc_mask=dc_mask,
        )
    return out


# Wrapper that matches process_batch_tiled's expected fn signature
def _histogram_match_image_fn(image_src: np.ndarray, image_tgt: np.ndarray, strength: float) -> np.ndarray:
    return _histogram_match_image(image_src, image_tgt, strength)


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class SpectralHistogramMatch:
    """
    Matches the FFT magnitude distribution of image_source to that of
    image_target — the frequency-domain equivalent of histogram matching.

    Unlike pixel-space histogram matching:
      • Colour is not directly modified (phase is preserved).
      • Spatial structure (edges, object positions) is preserved (phase).
      • Only the statistical distribution of spectral energy is transferred.

    Pre-processing use: make a synthetic/AI image spectrally indistinguishable
    from a natural photograph before VAE encoding.

    Post-processing use: give a generated image the same grain/texture character
    as the source without altering generated colours or content.
    """

    CATEGORY = "Spectral Preprocessing"

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_source": (
                    "IMAGE",
                    {"tooltip": "Image whose spectral magnitude distribution will be remapped."},
                ),
                "image_target": (
                    "IMAGE",
                    {"tooltip": "Reference image — its magnitude distribution is the target."},
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "0 = no change to source.  "
                            "1 = fully match target magnitude distribution.  "
                            "0.5–0.8 is a good starting range."
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
        image_source: "torch.Tensor",
        image_target: "torch.Tensor",
        strength: float,
        tile_size: int,
        tile_overlap: int,
    ) -> tuple:
        arr_src        = tensor_to_numpy(image_source)
        arr_tgt        = tensor_to_numpy(image_target)
        orig_h, orig_w = arr_src.shape[1], arr_src.shape[2]

        batch = arr_src.shape[0]
        out   = np.stack(
            [
                _histogram_match_image(
                    arr_src[i],
                    arr_tgt[i % arr_tgt.shape[0]],
                    strength=strength,
                )
                for i in range(batch)
            ],
            axis=0,
        )

        # Tiling is applied per-image if tile_size > 0
        if tile_size > 0:
            from ..utils.fft_utils import process_image_tiled
            out = np.stack(
                [
                    process_image_tiled(
                        arr_src[i],
                        _histogram_match_image_fn,
                        tile_size=tile_size,
                        tile_overlap=tile_overlap,
                        image_tgt=arr_tgt[i % arr_tgt.shape[0]],
                        strength=strength,
                    )
                    for i in range(batch)
                ],
                axis=0,
            )

        out = center_crop_to_match(out, orig_h, orig_w)
        return (numpy_to_tensor(clamp01(out)),)
