"""
Node 9 — Spectral Magnitude Blend
====================================
Blends the FFT magnitude of two images while preserving the phase of image A.

The classic insight from Portilla-Simoncelli and earlier Oppenheim & Lim work:

  • Magnitude  encodes "what kind of texture / spectral character"
  • Phase      encodes "where things are" (edges, structure, spatial layout)

Keeping phase_A and blending magnitudes produces an image that retains the
spatial structure of A but gradually adopts the spectral character of B.

Practical uses in the Flux img2img context:

  • Feed a known-clean natural photo as B and an AI-generated image as A.
    At blend=0.3–0.5 the output has A's content but B's spectral texture.
    This can make the AI image "feel" more like a real photograph to the VAE.

  • Blend two AI images where one has better high-frequency texture than the
    other.  Mix magnitudes to inherit the texture without changing composition.

  • Use a natural photo as a "spectral donor" to de-synthetic-ify an AI image
    before feeding it to the VAE encoder.

Algorithm:
  1.  FFT both images (per channel)
  2.  Extract mag_A, phase_A, mag_B, phase_B
  3.  Optionally resize B to match A if resolutions differ
  4.  Blended magnitude  =  (1 − blend) * mag_A  +  blend * mag_B
  5.  Reconstruct spectrum from blended_mag and phase_A
  6.  IFFT

A `frequency_mask` parameter lets you restrict the blend to a specific
radial band: 'all', 'low', 'mid', or 'high' — so you can, for example,
only adopt B's high-frequency texture while keeping A's low-frequency tones.
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
    center_crop_to_match,
    clamp01,
)


# Radial band boundaries (normalised radius)
_BAND_LIMITS = {
    "all":  (0.0, 1.0),
    "low":  (0.0, 0.15),
    "mid":  (0.15, 0.5),
    "high": (0.5, 1.0),
}


def _blend_magnitude_channel(
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
    mag_b  = magnitude(spec_b)
    ph_a   = phase(spec_a)

    # Build band mask
    band_mask = (r_norm >= band_lo) & (r_norm < band_hi)

    # Blend magnitudes within the selected band; keep mag_a outside
    new_mag              = mag_a.copy()
    new_mag[band_mask]   = (1.0 - blend) * mag_a[band_mask] + blend * mag_b[band_mask]

    new_spec = reconstruct(new_mag, ph_a)
    return ifft2_channel(new_spec)


def _blend_magnitude_image(
    image_a: np.ndarray,
    image_b: np.ndarray,
    blend: float,
    frequency_band: str,
) -> np.ndarray:
    h, w, c = image_a.shape

    # Resize B to match A if needed
    image_b_r = resize_to_match(image_b, h, w)

    r_norm            = radial_grid(h, w)
    band_lo, band_hi  = _BAND_LIMITS.get(frequency_band, (0.0, 1.0))

    out = np.empty_like(image_a)
    for ch in range(c):
        out[:, :, ch] = _blend_magnitude_channel(
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

class SpectralMagnitudeBlend:
    """
    Blends the FFT magnitudes of two images while retaining the phase (spatial
    structure) of image A.

    At blend=0 the output is identical to A.
    At blend=1 the output has A's edges and layout but B's spectral texture.

    Use image_b as a "spectral donor" — e.g. a clean natural photograph —
    to import a more natural spectral character into an AI-generated image
    before VAE encoding.

    The frequency_band parameter limits blending to a specific radial band
    (low / mid / high) so you can target only the problematic part of the
    spectrum.
    """

    CATEGORY = "Spectral Preprocessing"

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "Primary image — phase is always taken from here."}),
                "image_b": ("IMAGE", {"tooltip": "Spectral donor — magnitude is blended from here."}),
                "blend": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "0 = keep A's magnitude entirely, 1 = use B's magnitude entirely.",
                    },
                ),
                "frequency_band": (
                    ["all", "low", "mid", "high"],
                    {
                        "default": "all",
                        "tooltip": (
                            "Restrict blending to a radial frequency band.  "
                            "'all' blends the full spectrum.  "
                            "'high' only adopts B's high-frequency texture."
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
                _blend_magnitude_image(
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
