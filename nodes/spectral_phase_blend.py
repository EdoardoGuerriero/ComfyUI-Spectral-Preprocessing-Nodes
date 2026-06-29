"""
Node 10 — Spectral Phase Blend
=================================
Blends the FFT phase of two images while retaining the magnitude of image A.

The complement of SpectralMagnitudeBlend.

Phase encodes spatial structure: edges, positions of objects, texture
micro-geometry.  Blending phases morphs the spatial structure of the image
while keeping its spectral character (overall texture roughness, energy
distribution) fixed.

Practical uses:

  • Structure morphing / dream transitions between two images.
  • Inject the spatial micro-structure of image B into image A's "envelope".
    This is subtler than magnitude blending: the output still looks like A,
    but its fine-detail spatial layout gradually adopts B's geometry.
  • Frequency-domain content injection: use B as a guide for how edges and
    textures *should* be positioned, while A determines the spectral energy.

⚠  Phase blending is perceptually much more dramatic than magnitude blending
   even at small blend values.  Start at 0.05–0.1 and increase carefully.

Algorithm:
  1.  FFT both images (per channel)
  2.  Extract mag_A, phase_A, phase_B
  3.  Circular interpolation of phases:
        phase_blend = angle( (1−t)·exp(i·phase_A) + t·exp(i·phase_B) )
      This correctly handles the ±π wrap without artifacts.
  4.  Reconstruct spectrum from mag_A and phase_blend
  5.  IFFT

A `frequency_band` parameter lets you restrict phase blending to low, mid,
or high frequencies.  Low-frequency phase blending is particularly
interesting: it shifts where large structures sit while preserving texture.
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
    lerp_phase,
    resize_to_match,
    center_crop_to_match,
    clamp01,
)


_BAND_LIMITS = {
    "all":  (0.0, 1.0),
    "low":  (0.0, 0.15),
    "mid":  (0.15, 0.5),
    "high": (0.5, 1.0),
}


def _blend_phase_channel(
    ch_a: np.ndarray,
    ch_b: np.ndarray,
    blend: float,
    r_norm: np.ndarray,
    band_lo: float,
    band_hi: float,
) -> np.ndarray:
    spec_a = fft2_channel(ch_a)
    spec_b = fft2_channel(ch_b)

    mag_a  = magnitude(spec_a)
    ph_a   = phase(spec_a)
    ph_b   = phase(spec_b)

    band_mask = (r_norm >= band_lo) & (r_norm < band_hi)

    # Circular phase interpolation inside the band
    ph_mixed          = ph_a.copy()
    ph_mixed[band_mask] = lerp_phase(ph_a[band_mask], ph_b[band_mask], blend)

    new_spec = reconstruct(mag_a, ph_mixed)
    return ifft2_channel(new_spec)


def _blend_phase_image(
    image_a: np.ndarray,
    image_b: np.ndarray,
    blend: float,
    frequency_band: str,
) -> np.ndarray:
    h, w, c   = image_a.shape
    image_b_r = resize_to_match(image_b, h, w)

    r_norm           = radial_grid(h, w)
    band_lo, band_hi = _BAND_LIMITS.get(frequency_band, (0.0, 1.0))

    out = np.empty_like(image_a)
    for ch in range(c):
        out[:, :, ch] = _blend_phase_channel(
            image_a[:, :, ch],
            image_b_r[:, :, ch],
            blend=blend,
            r_norm=r_norm,
            band_lo=band_lo,
            band_hi=band_hi,
        )
    return out


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class SpectralPhaseBlend:
    """
    Blends the FFT phase of two images while retaining the magnitude
    (spectral energy distribution) of image A.

    Phase encodes spatial structure — where edges and textures are positioned.
    Blending phases morphs the spatial geometry of the image while its
    spectral character (energy distribution, overall texture roughness) is
    preserved from A.

    ⚠  Even small blend values (0.05–0.1) produce visible structural changes.
    The effect is much more dramatic than magnitude blending.

    The frequency_band parameter lets you morph only low-frequency structure
    (large shapes), only mid-frequency geometry, or only fine-detail texture.
    """

    CATEGORY = "Spectral Preprocessing"

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "Primary image — magnitude is always taken from here."}),
                "image_b": ("IMAGE", {"tooltip": "Structure donor — phase is blended from here."}),
                "blend": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.005,
                        "display": "slider",
                        "tooltip": (
                            "0 = keep A's phase entirely, 1 = use B's phase entirely.  "
                            "Start at 0.05 and increase carefully — phase blending is "
                            "perceptually strong even at low values."
                        ),
                    },
                ),
                "frequency_band": (
                    ["all", "low", "mid", "high"],
                    {
                        "default": "all",
                        "tooltip": (
                            "'low'  → morph large shapes / global structure.  "
                            "'mid'  → morph medium textures and object edges.  "
                            "'high' → morph fine-detail micro-texture only."
                        ),
                    },
                ),
            }
        }

    def apply(
        self,
        image_a: "torch.Tensor",
        image_b: "torch.Tensor",
        blend: float,
        frequency_band: str,
    ) -> tuple:
        arr_a          = tensor_to_numpy(image_a)
        arr_b          = tensor_to_numpy(image_b)
        orig_h, orig_w = arr_a.shape[1], arr_a.shape[2]

        out = np.stack(
            [
                _blend_phase_image(
                    arr_a[i],
                    arr_b[i % arr_b.shape[0]],
                    blend=blend,
                    frequency_band=frequency_band,
                )
                for i in range(arr_a.shape[0])
            ],
            axis=0,
        )
        out = center_crop_to_match(out, orig_h, orig_w)
        return (numpy_to_tensor(clamp01(out)),)
