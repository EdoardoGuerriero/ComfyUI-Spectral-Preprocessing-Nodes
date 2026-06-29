"""
Node 14 — Spectral Angular Equalizer
======================================
Equalizes the angular energy distribution of the FFT spectrum toward the
isotropic mean, suppressing systematic directional biases.

Distinction from DirectionalArtifactSuppressor
-----------------------------------------------
DirectionalArtifactSuppressor (Node 4) targets *isolated* directional spikes —
narrow angular regions where energy anomalously exceeds the local angular mean.
It is an AC-level outlier suppressor (analogous to a limiter).

SpectralAngularEqualizer targets *systematic* directional imbalances — the
entire angular distribution is skewed, not just a few narrow spikes.  It is a
DC-level normalizer (analogous to a gain equalizer).

Typical sources of systematic directional bias
-----------------------------------------------
  • Images dominated by horizontal or vertical textures (tiles, screens,
    printed media, fabric, architecture) have excess energy at 0° and 90°.
  • Scanner line artifacts: excess energy at 0° (horizontal scan lines).
  • Codec / compression artifacts: JPEG 8×8 block grid → excess at 0° / 90°.
  • Lens or sensor defects: vignetting, micro-lens array patterns.
  • AI-generated content: diffusion models often introduce directional
    frequency preferences not present in natural photographs.

Algorithm
---------
  1.  FFT(channel) → mag, phase
  2.  Optionally protect near-DC (small radii carry structural anisotropy that
      reflects actual image content, not processing bias).
  3.  Bin the spectrum into N_BINS angular sectors [0, π).
      (Magnitude spectrum has 180° symmetry: F(−f) = F*(f), so only 0→π
       is independent.)
  4.  For each sector, compute mean magnitude E(θ).
  5.  Compute the isotropic target T = mean(E(θ)) over all sectors.
  6.  Per-sector gain G(θ) = (T / E(θ))^equalize_power
      — equalize_power=1: full normalization to isotropic mean.
      — equalize_power=0.5: half-way (softer, less aggressive).
      — equalize_power=0: no effect.
      Gains are clamped to [min_gain, max_gain] to prevent extreme corrections.
  7.  Apply G(θ) to each coefficient based on its angular sector.
  8.  Blend with original:  mag_out = (1 − strength) · mag + strength · mag_eq
  9.  Protect DC coefficient.
  10. Reconstruct from mag_out and original phase → IFFT.

The gain is applied symmetrically (G(θ) = G(θ+π)) because of Hermitian symmetry.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..utils.fft_utils import (
    tensor_to_numpy,
    numpy_to_tensor,
    fft2_channel,
    ifft2_channel,
    magnitude,
    phase,
    reconstruct,
    radial_grid,
    angular_grid,
    get_dc_mask,
    process_batch_tiled,
    center_crop_to_match,
    clamp01,
)


# ---------------------------------------------------------------------------
# Core algorithm (single channel)
# ---------------------------------------------------------------------------

def _angular_equalize_channel(
    channel: np.ndarray,
    strength: float,
    n_angle_bins: int,
    equalize_power: float,
    min_gain: float,
    max_gain: float,
    protect_radius: float,
    r_norm: np.ndarray,
    ang: np.ndarray,
    dc_mask: np.ndarray,
) -> np.ndarray:
    spec = fft2_channel(channel)
    mag  = magnitude(spec)
    ph   = phase(spec)

    # Mask for coefficients that are eligible for equalization
    active = r_norm >= protect_radius

    # Angular bin index for every pixel [0, n_angle_bins)
    bin_idx = np.floor(ang / np.pi * n_angle_bins).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_angle_bins - 1)

    # Mean magnitude per angular bin (active pixels only)
    bin_energy = np.zeros(n_angle_bins, dtype=np.float64)
    bin_count  = np.zeros(n_angle_bins, dtype=np.int64)

    for b in range(n_angle_bins):
        mask = active & (bin_idx == b)
        if mask.any():
            bin_energy[b] = mag[mask].mean()
            bin_count[b]  = mask.sum()

    # Isotropic target = mean energy over non-empty bins
    nonempty = bin_count > 0
    if not nonempty.any():
        return ifft2_channel(spec)

    target = bin_energy[nonempty].mean()
    if target < 1e-12:
        return ifft2_channel(spec)

    # Per-bin gain
    gain_per_bin = np.ones(n_angle_bins, dtype=np.float64)
    for b in range(n_angle_bins):
        if bin_count[b] > 0 and bin_energy[b] > 1e-12:
            raw_gain = (target / bin_energy[b]) ** equalize_power
            gain_per_bin[b] = np.clip(raw_gain, min_gain, max_gain)

    # Smooth the gain curve across angles to avoid discontinuities at bin edges
    smooth_bins = max(1, n_angle_bins // 12)
    gain_per_bin = uniform_filter1d(gain_per_bin, size=smooth_bins, mode="wrap")

    # Build per-pixel gain map
    gain_map = np.ones_like(mag, dtype=np.float64)
    for b in range(n_angle_bins):
        mask = active & (bin_idx == b)
        gain_map[mask] = gain_per_bin[b]

    mag_eq  = mag * gain_map
    mag_out = (1.0 - strength) * mag + strength * mag_eq

    # DC protection
    mag_out[dc_mask] = mag[dc_mask]

    return ifft2_channel(reconstruct(mag_out, ph))


# ---------------------------------------------------------------------------
# Per-image wrapper
# ---------------------------------------------------------------------------

def _angular_equalize_image(
    image: np.ndarray,
    strength: float,
    n_angle_bins: int,
    equalize_power: float,
    min_gain: float,
    max_gain: float,
    protect_radius: float,
) -> np.ndarray:
    h, w, c = image.shape
    dc_mask = get_dc_mask(h, w, radius=2)
    r_norm  = radial_grid(h, w)
    ang     = angular_grid(h, w)

    out = np.empty_like(image)
    for ch in range(c):
        out[:, :, ch] = _angular_equalize_channel(
            image[:, :, ch],
            strength=strength,
            n_angle_bins=n_angle_bins,
            equalize_power=equalize_power,
            min_gain=min_gain,
            max_gain=max_gain,
            protect_radius=protect_radius,
            r_norm=r_norm,
            ang=ang,
            dc_mask=dc_mask,
        )
    return out


def _angular_equalize_image_fn(image: np.ndarray, **kwargs) -> np.ndarray:
    return _angular_equalize_image(image, **kwargs)


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class SpectralAngularEqualizer:
    """
    Equalizes systematic directional biases in the FFT spectrum.

    Unlike DirectionalArtifactSuppressor (which targets isolated spikes),
    this node normalizes the *entire* angular energy distribution toward the
    isotropic mean.  It corrects for:

      • Horizontal/vertical texture bias (architecture, screens, fabric)
      • Scanner line artifacts
      • JPEG block-grid artifacts (energy peaks at 0° and 90°)
      • AI-generation directional frequency preferences

    The equalize_power parameter controls aggressiveness:
      • 1.0 = full normalization — every angle gets the same mean energy
      • 0.5 = half-way correction — softer, preserves more of the original bias
      • 0.0 = no effect (identity)

    Gains are clamped and the gain curve is smoothed across angles to avoid
    sharp discontinuities between adjacent bins.
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
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Blend between original (0) and fully equalized (1).",
                    },
                ),
                "equalize_power": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": (
                            "Exponent on the per-bin gain.  "
                            "1.0 = full normalization (all angles equal energy).  "
                            "0.5 = square-root gain (softer).  "
                            "0.0 = no gain (identity).  "
                            "Start at 0.5–0.75."
                        ),
                    },
                ),
                "n_angle_bins": (
                    "INT",
                    {
                        "default": 36,
                        "min": 8,
                        "max": 180,
                        "step": 4,
                        "tooltip": (
                            "Number of angular sectors in [0°, 180°).  "
                            "36 = 5° per bin.  More bins = finer resolution "
                            "but noisier statistics.  36–72 is a good range."
                        ),
                    },
                ),
                "protect_radius": (
                    "FLOAT",
                    {
                        "default": 0.08,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Normalised radius below which coefficients are excluded "
                            "from equalization.  Low-frequency energy reflects real image "
                            "anisotropy (e.g. a landscape is not isotropic at low freq).  "
                            "0.05–0.15 is recommended."
                        ),
                    },
                ),
                "min_gain": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Minimum per-angle gain.  Prevents over-amplifying very "
                            "quiet directions.  0.1 means the quietest direction "
                            "can be boosted by at most 10×."
                        ),
                    },
                ),
                "max_gain": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 1.0,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "Maximum per-angle gain.  Prevents extreme suppression of dominant directions.",
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
        equalize_power: float,
        n_angle_bins: int,
        protect_radius: float,
        min_gain: float,
        max_gain: float,
        tile_size: int,
        tile_overlap: int,
    ) -> tuple:
        arr            = tensor_to_numpy(image)
        orig_h, orig_w = arr.shape[1], arr.shape[2]

        kwargs = dict(
            strength=strength,
            n_angle_bins=n_angle_bins,
            equalize_power=equalize_power,
            min_gain=min_gain,
            max_gain=max_gain,
            protect_radius=protect_radius,
        )

        out = process_batch_tiled(arr, _angular_equalize_image_fn, tile_size, tile_overlap, **kwargs)
        out = center_crop_to_match(out, orig_h, orig_w)
        return (numpy_to_tensor(clamp01(out)),)
