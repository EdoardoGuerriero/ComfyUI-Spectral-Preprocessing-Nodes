"""
Node 4 — Directional Artifact Suppressor
==========================================
Detects and attenuates dominant directional spikes in the Fourier domain.

Many artifact types produce energy that is confined to specific orientations
in frequency space:

  • Checkerboard patterns (transposed convolutions)  → diagonal spikes at ±45°
  • ESRGAN / upscaling grid artifacts               → horizontal + vertical
  • CNN periodic patterns                            → multiples of 90°
  • Aliasing from downsampling                       → axis-aligned bands
  • Compression grid (JPEG 8×8)                     → peaks at multiples of 8px

Algorithm:
  1.  FFT → magnitude
  2.  Convert to polar coordinates (r, θ) via a precomputed map
  3.  Compute the angular energy distribution  E(θ)  by summing magnitude²
      over all radii above a minimum (exclude DC neighbourhood)
  4.  Estimate smooth angular background via circular median filter
  5.  Compute residual z-score per angle
  6.  Build a smooth angular attenuation mask M(θ)
  7.  Map M back to Cartesian Fourier space
  8.  Protect a small radial band around DC regardless of angle
  9.  Apply mask to magnitude, reconstruct, IFFT

The attenuation mask is always radially symmetric for DC (r < dc_radius),
preventing any change to the mean image brightness.

Note on angular resolution:
  The Fourier magnitude has 180° periodicity (F(−r) = F*(r)), so we work
  in [0°, 180°) and mirror the mask to [180°, 360°) before applying.
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
    angular_grid,
    radial_grid,
    get_dc_mask,
    stable_sigmoid,
    process_batch_tiled,
    center_crop_to_match,
    clamp01,
)


# ---------------------------------------------------------------------------
# Core algorithm (single channel)
# ---------------------------------------------------------------------------

def _suppress_directional_channel(
    channel: np.ndarray,
    threshold_sigma: float,
    attenuation: float,
    angular_kernel: int,
    min_radius: float,
    dc_mask: np.ndarray,
    angles: np.ndarray,
    r_norm: np.ndarray,
    n_angle_bins: int,
) -> np.ndarray:
    """
    Suppress directional artifacts for a single [H, W] channel.

    Parameters
    ----------
    channel         : [H, W] float32
    threshold_sigma : z-score threshold for anomalous angular energy
    attenuation     : max fraction of directional energy to suppress
    angular_kernel  : size of the circular median filter in angle bins
    min_radius      : minimum normalised radius included in angular analysis
    dc_mask         : [H, W] bool — pixels always left at gain=1
    angles          : [H, W] float32, angle in [0, π)
    r_norm          : [H, W] float32, normalised radial distance
    n_angle_bins    : angular resolution for the analysis
    """
    spectrum = fft2_channel(channel)
    mag      = magnitude(spectrum)
    ph       = phase(spectrum)

    # Exclude pixels too close to DC from the analysis ring
    analysis_ring = r_norm >= min_radius

    # --- angular energy profile -------------------------------------------
    bin_edges   = np.linspace(0.0, np.pi, n_angle_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    energy      = np.zeros(n_angle_bins, dtype=np.float64)

    for i in range(n_angle_bins):
        mask = analysis_ring & (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        if mask.any():
            energy[i] = np.sum(mag[mask] ** 2)

    # --- smooth background via circular median ----------------------------
    # Wrap the array for circular filtering
    k = angular_kernel
    padded      = np.concatenate([energy[-k:], energy, energy[:k]])
    background  = np.array([
        np.median(padded[i: i + 2 * k + 1]) for i in range(n_angle_bins)
    ])

    # --- z-score per angle -----------------------------------------------
    residual   = energy - background
    mad        = np.median(np.abs(residual - np.median(residual)))
    local_std  = mad * 1.4826 + 1e-8
    z          = residual / local_std

    # --- soft attenuation mask per angle bin -----------------------------
    overshoot      = z - threshold_sigma
    sigmoid        = stable_sigmoid(overshoot * 2.0)
    gain_per_bin   = 1.0 - attenuation * sigmoid
    gain_per_bin   = np.clip(gain_per_bin, 1.0 - attenuation, 1.0)

    # --- map gain to every pixel via its angle ---------------------------
    # np.interp requires sorted x; bin_centers is already sorted and in [0, π)
    gain_2d = np.interp(
        angles.ravel(), bin_centers, gain_per_bin
    ).reshape(angles.shape).astype(np.float32)

    # For pixels not in the analysis ring (very low frequencies), gain = 1
    gain_2d[~analysis_ring] = 1.0

    # DC protection
    gain_2d[dc_mask] = 1.0

    # --- apply and reconstruct -------------------------------------------
    new_mag  = mag * gain_2d
    new_spec = reconstruct(new_mag, ph)
    return ifft2_channel(new_spec)


# ---------------------------------------------------------------------------
# Per-image wrapper
# ---------------------------------------------------------------------------

def _suppress_directional_image(
    image: np.ndarray,
    threshold_sigma: float,
    attenuation: float,
    angular_kernel: int,
    min_radius: float,
    n_angle_bins: int,
) -> np.ndarray:
    h, w, c  = image.shape
    dc_mask  = get_dc_mask(h, w, radius=3)
    angles   = angular_grid(h, w)
    r_norm   = radial_grid(h, w)

    out = np.empty_like(image)
    for ch in range(c):
        out[:, :, ch] = _suppress_directional_channel(
            image[:, :, ch],
            threshold_sigma=threshold_sigma,
            attenuation=attenuation,
            angular_kernel=angular_kernel,
            min_radius=min_radius,
            dc_mask=dc_mask,
            angles=angles,
            r_norm=r_norm,
            n_angle_bins=n_angle_bins,
        )
    return out


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class DirectionalArtifactSuppressor:
    """
    Detects and attenuates frequency-domain energy that is anomalously
    concentrated along specific angular directions.

    Targets: checkerboards, CNN/upscaling grid patterns, aliasing bands,
    ESRGAN stripe artifacts, and any periodic texture with a dominant orientation.

    The analysis is performed in the angular domain of the FFT magnitude —
    a bin-wise z-score flags directions that carry significantly more energy
    than their neighbours, and a smooth gain mask suppresses only those directions.
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
                        "max": 15.0,
                        "step": 0.1,
                        "display": "slider",
                        "tooltip": (
                            "Z-score threshold for flagging an angular direction as "
                            "anomalous.  Lower = more aggressive."
                        ),
                    },
                ),
                "attenuation": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Maximum fraction of energy removed in flagged directions.",
                    },
                ),
                "angular_kernel": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 30,
                        "step": 1,
                        "tooltip": (
                            "Half-width (in angle bins) of the circular median filter "
                            "used to estimate the angular background.  Larger = smoother "
                            "background, less sensitive to broad directional biases."
                        ),
                    },
                ),
                "min_radius": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 0.4,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Minimum normalised radius included in the angular analysis. "
                            "Pixels closer to DC than this are always left unmodified."
                        ),
                    },
                ),
                "n_angle_bins": (
                    "INT",
                    {
                        "default": 180,
                        "min": 36,
                        "max": 720,
                        "step": 36,
                        "tooltip": (
                            "Angular resolution of the analysis (number of bins over "
                            "0°–180°).  More bins = finer directional discrimination."
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
        angular_kernel: int,
        min_radius: float,
        n_angle_bins: int,
        tile_size: int,
        tile_overlap: int,
    ) -> tuple:
        arr            = tensor_to_numpy(image)
        orig_h, orig_w = arr.shape[1], arr.shape[2]
        processed      = process_batch_tiled(
            arr,
            _suppress_directional_image,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            threshold_sigma=threshold_sigma,
            attenuation=attenuation,
            angular_kernel=angular_kernel,
            min_radius=min_radius,
            n_angle_bins=n_angle_bins,
        )
        processed = center_crop_to_match(processed, orig_h, orig_w)
        return (numpy_to_tensor(clamp01(processed)),)
