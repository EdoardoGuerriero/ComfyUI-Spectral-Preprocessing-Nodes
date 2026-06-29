"""
Node 12 — Radial Stratified Histogram Match
==============================================
The frequency-domain equivalent of CLAHE (Contrast-Limited Adaptive
Histogram Equalisation) — performs histogram matching of FFT magnitudes
independently within each radial frequency band.

Why stratify by radius?
-----------------------
The FFT magnitude distribution is strongly non-stationary across spatial
frequencies:
  • Low-frequency coefficients (near DC) have very large magnitudes —
    they carry the global illumination and colour.
  • High-frequency coefficients have small magnitudes — they carry fine
    texture and noise.

Applying a single global CDF mapping (as in SpectralHistogramMatch) mixes
these two regimes: a low-frequency value that ranks at the 90th percentile
globally might only rank at the 30th percentile within its own frequency band.
The global mapping therefore transfers the wrong statistics.

Per-band matching (radial stratification) solves this:
  1.  Divide the spectrum into N concentric annular rings.
  2.  Within each ring, match the CDF of source magnitudes to target magnitudes
      independently.
  3.  Recombine and apply.

This is strictly more accurate than global matching because it respects the
statistical structure of natural images — a different CDF per band.

Comparison with existing nodes
-------------------------------
  Node                          | What it matches
  ------------------------------|------------------------------------------
  RadialSpectrumNormalizer      | Radial mean power (1-D curve)
  SpectralHistogramMatch        | Global magnitude CDF (1 CDF for all bands)
  RadialStratifiedHistogramMatch| Per-band magnitude CDF (N independent CDFs)

This node is the most expressive of the three and also the most data-driven
— it requires no target curve assumptions (unlike the 1/f^alpha model).

Tiling note
-----------
When tiling is active, each tile defines its own DC centre and therefore its
own radial grid.  The per-band statistics are therefore computed per-tile
rather than globally.  This is appropriate for local texture matching but
will not match the global spectral balance of the image.  For global spectral
matching, leave tile_size = 0.

Algorithm (per channel)
-----------------------
  1.  FFT(source) → mag_src, phase_src
  2.  FFT(target) → mag_tgt  (resized to source resolution for the FFT)
  3.  Build radial grid (normalised radius 0–1)
  4.  For each radial band [r_lo, r_hi]:
        a.  Extract source pixels in band → src_band
        b.  Extract target pixels in band → tgt_band
        c.  Skip band if either is empty or too small
        d.  Rank-match src_band to tgt_band distribution
        e.  Write matched values back into the magnitude array
  5.  Optionally protect the innermost band (near-DC) at gain=1
  6.  Blend:  mag_out = (1 − strength) * mag_src + strength * mag_matched
  7.  Protect DC coefficient exactly
  8.  Reconstruct from mag_out and phase_src → IFFT
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
    resize_to_match,
    get_dc_mask,
    process_batch_tiled,
    center_crop_to_match,
    clamp01,
)
from .spectral_histogram_match import _rank_match_magnitudes


# ---------------------------------------------------------------------------
# Core algorithm (single channel)
# ---------------------------------------------------------------------------

def _stratified_match_channel(
    ch_src: np.ndarray,
    ch_tgt: np.ndarray,
    strength: float,
    n_bands: int,
    preserve_low_freq: float,
    dc_mask: np.ndarray,
    r_norm: np.ndarray,
) -> np.ndarray:
    """
    Apply radially-stratified histogram matching to a single [H, W] channel.

    Parameters
    ----------
    ch_src            : [H, W] float32 source channel
    ch_tgt            : [H, W] float32 target channel (same resolution as src)
    strength          : blend factor (0 = no change, 1 = full match)
    n_bands           : number of radial annuli
    preserve_low_freq : normalised radius below which matching is skipped
    dc_mask           : [H, W] bool — DC pixels are never modified
    r_norm            : [H, W] float32 normalised radial distances
    """
    spec_src = fft2_channel(ch_src)
    spec_tgt = fft2_channel(ch_tgt)

    mag_src = magnitude(spec_src)
    mag_tgt = magnitude(spec_tgt)
    ph_src  = phase(spec_src)

    mag_matched = mag_src.copy()

    # Build band edges on the normalised radius axis
    edges = np.linspace(0.0, 1.0, n_bands + 1)

    for i in range(n_bands):
        r_lo = edges[i]
        r_hi = edges[i + 1]

        # Skip bands entirely within the low-frequency protection zone
        if r_hi <= preserve_low_freq:
            continue

        band_mask = (r_norm >= r_lo) & (r_norm < r_hi)

        src_band = mag_src[band_mask]
        tgt_band = mag_tgt[band_mask]

        # Need at least a handful of coefficients to compute a meaningful CDF
        if len(src_band) < 4 or len(tgt_band) < 4:
            continue

        matched_band = _rank_match_magnitudes(src_band.reshape(1, -1), tgt_band.reshape(1, -1))
        mag_matched[band_mask] = matched_band.ravel()

    # Blend
    mag_out = (1.0 - strength) * mag_src + strength * mag_matched

    # DC protection
    mag_out[dc_mask] = mag_src[dc_mask]

    new_spec = reconstruct(mag_out.astype(np.float64), ph_src)
    return ifft2_channel(new_spec)


# ---------------------------------------------------------------------------
# Per-image wrapper
# ---------------------------------------------------------------------------

def _stratified_match_image(
    image_src: np.ndarray,
    image_tgt: np.ndarray,
    strength: float,
    n_bands: int,
    preserve_low_freq: float,
) -> np.ndarray:
    h, w, c = image_src.shape
    dc_mask = get_dc_mask(h, w, radius=2)
    r_norm  = radial_grid(h, w)

    tgt_r = resize_to_match(image_tgt, h, w)

    out = np.empty_like(image_src)
    for ch in range(c):
        out[:, :, ch] = _stratified_match_channel(
            image_src[:, :, ch],
            tgt_r[:, :, ch],
            strength=strength,
            n_bands=n_bands,
            preserve_low_freq=preserve_low_freq,
            dc_mask=dc_mask,
            r_norm=r_norm,
        )
    return out


def _stratified_match_image_fn(
    image_src: np.ndarray,
    image_tgt: np.ndarray,
    strength: float,
    n_bands: int,
    preserve_low_freq: float,
) -> np.ndarray:
    return _stratified_match_image(image_src, image_tgt, strength, n_bands, preserve_low_freq)


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class RadialStratifiedHistogramMatch:
    """
    Matches the FFT magnitude distribution of image_source to image_target
    independently within each radial frequency band.

    This is the frequency-domain equivalent of CLAHE — by matching CDFs per
    band rather than globally, it correctly handles the fact that low- and
    high-frequency coefficients follow very different statistical distributions.

    Result: the output has the same per-band spectral texture as the target,
    while preserving the source's spatial structure (edges, content, colours)
    via the unchanged phase.

    More expressive than SpectralHistogramMatch (global) and
    RadialSpectrumNormalizer (mean-only) — this matches the full statistical
    shape at every frequency scale independently.
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
                    {"tooltip": "Image whose per-band spectral distribution will be remapped."},
                ),
                "image_target": (
                    "IMAGE",
                    {"tooltip": "Reference — its per-band magnitude distribution is the target."},
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "0 = no change, 1 = fully match target per-band distribution.",
                    },
                ),
                "n_bands": (
                    "INT",
                    {
                        "default": 16,
                        "min": 2,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Number of radial frequency bands.  More bands = finer "
                            "per-scale control but more sensitive to noise in the "
                            "statistics.  8–24 is a good range."
                        ),
                    },
                ),
                "preserve_low_freq": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Normalised radius below which bands are skipped entirely.  "
                            "Protects global brightness and colour from being matched away.  "
                            "Keep at 0.05–0.1 for post-processing; raise to 0.15–0.2 for "
                            "pre-processing."
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
                        "tooltip": (
                            "Tile size for large images. 0 = whole image (recommended "
                            "for this node — tiling changes the band statistics)."
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
        n_bands: int,
        preserve_low_freq: float,
        tile_size: int,
        tile_overlap: int,
    ) -> tuple:
        arr_src        = tensor_to_numpy(image_source)
        arr_tgt        = tensor_to_numpy(image_target)
        orig_h, orig_w = arr_src.shape[1], arr_src.shape[2]

        batch = arr_src.shape[0]

        if tile_size > 0:
            from ..utils.fft_utils import process_image_tiled
            out = np.stack(
                [
                    process_image_tiled(
                        arr_src[i],
                        _stratified_match_image_fn,
                        tile_size=tile_size,
                        tile_overlap=tile_overlap,
                        image_tgt=arr_tgt[i % arr_tgt.shape[0]],
                        strength=strength,
                        n_bands=n_bands,
                        preserve_low_freq=preserve_low_freq,
                    )
                    for i in range(batch)
                ],
                axis=0,
            )
        else:
            out = np.stack(
                [
                    _stratified_match_image(
                        arr_src[i],
                        arr_tgt[i % arr_tgt.shape[0]],
                        strength=strength,
                        n_bands=n_bands,
                        preserve_low_freq=preserve_low_freq,
                    )
                    for i in range(batch)
                ],
                axis=0,
            )

        out = center_crop_to_match(out, orig_h, orig_w)
        return (numpy_to_tensor(clamp01(out)),)
