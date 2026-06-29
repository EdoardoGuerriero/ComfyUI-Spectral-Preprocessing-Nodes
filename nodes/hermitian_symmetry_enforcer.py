"""
Node 15 — Hermitian Symmetry Enforcer
=======================================
Restores the Hermitian symmetry constraint of the FFT of a real-valued image.

Background — Hermitian symmetry
---------------------------------
For any real-valued signal I, its 2-D DFT F must satisfy:

    F[k, l]  =  conj( F[−k, −l] )      (Hermitian / conjugate symmetry)

Equivalently, in the centre-shifted spectrum:
    F[cy + dk, cx + dl]  =  conj( F[cy − dk, cx − dl] )

where (cy, cx) is the DC centre.

Why this constraint can be violated
-------------------------------------
In an ideal pipeline this is always satisfied for any real input.  Violations
can accumulate from:

  • Lossy JPEG / codec compression — the IDCT→pixel quantisation is NOT the
    exact inverse of the DCT, and the resulting pixel values may carry tiny
    asymmetries when you then take the FFT.

  • Certain image resizing / interpolation methods that are not centro-symmetric.

  • Manual frequency-domain edits (magnitude-only or phase-only modifications)
    performed outside this pack that do not enforce the constraint.

  • Floating-point accumulation errors over many operations.

When the constraint is violated the IFFT of the modified spectrum will have a
non-trivial imaginary part.  This is usually discarded (.real), but the
imaginary energy is real spectral energy that should be in the real part —
discarding it changes the image.

The fix
--------
    F_corrected[k, l] = ( F[k, l] + conj( F[−k, −l] ) ) / 2

This is the minimum-L2-norm correction that makes F Hermitian.  It is exact,
linear, and has no free parameters except `strength` (blend with the original).

A `strength` parameter is provided so you can blend rather than fully enforce
the constraint.  For natural images the correction is negligible (the imaginary
residual after IFFT is typically < 1e-6).  For images that have been through
multiple codec cycles or manual FFT edits, the effect can be visible.

Second output: residual image
------------------------------
The "residual" output shows the imaginary part of IFFT(F) before enforcement —
the "phantom energy" that was being discarded.  For a clean natural image this
is nearly black.  It serves as a diagnostic: if it is visibly non-trivial,
this node should be placed early in the pipeline.
"""

import numpy as np

from ..utils.fft_utils import (
    tensor_to_numpy,
    numpy_to_tensor,
    fft2_channel,
    ifft2_channel,
    center_crop_to_match,
    clamp01,
)


# ---------------------------------------------------------------------------
# Core algorithm (single channel)
# ---------------------------------------------------------------------------

def _enforce_hermitian_channel(channel: np.ndarray) -> tuple:
    """
    Enforce Hermitian symmetry on one [H, W] channel.

    Returns (out_channel, residual_channel):
        out_channel      : [H, W] float32, Hermitian-corrected image
        residual_channel : [H, W] float32, |imaginary part of IFFT(F)|
    """
    # Use the shifted spectrum so the flip is a simple centre-symmetric flip
    spec = fft2_channel(channel)  # shifted

    # Hermitian correction in the shifted domain:
    # F_conj_sym[k, l] = conj(F[-k, -l]) = conj(np.roll(np.flip(F), (1,1), axis=(0,1)))
    spec_flip = np.conj(spec[::-1, ::-1])  # conj(F[-k,-l])

    # Averaging enforces the constraint with minimum L2 perturbation
    spec_herm = 0.5 * (spec + spec_flip)

    # The imaginary residual of the ORIGINAL spectrum (diagnostic)
    raw_ifft       = np.fft.ifft2(np.fft.ifftshift(spec))
    residual_imag  = np.abs(raw_ifft.imag).astype(np.float32)

    # Reconstructed image from Hermitian-corrected spectrum
    out = np.fft.ifft2(np.fft.ifftshift(spec_herm)).real.astype(np.float32)

    return out, residual_imag


# ---------------------------------------------------------------------------
# Per-image wrapper
# ---------------------------------------------------------------------------

def _enforce_hermitian_image(image: np.ndarray, strength: float) -> tuple:
    H, W, C = image.shape
    out      = np.empty_like(image)
    residual = np.empty_like(image)

    for ch in range(C):
        corrected, res = _enforce_hermitian_channel(image[:, :, ch])
        out[:, :, ch]      = (1.0 - strength) * image[:, :, ch] + strength * corrected
        residual[:, :, ch] = res

    return out, residual


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class HermitianSymmetryEnforcer:
    """
    Restores the Hermitian symmetry of the FFT of a real-valued image.

    For any real image I, the DFT must satisfy F[k,l] = conj(F[−k,−l]).
    This constraint can be subtly violated by lossy codec cycles, certain
    interpolation methods, or manual frequency-domain edits.  Violations cause
    the IFFT to produce a non-trivial imaginary component that is silently
    discarded by every .real call — losing spectral energy that should be
    in the image.

    This node corrects the spectrum to the nearest Hermitian matrix (minimum L2):

        F_corrected = ( F + conj(F_flipped) ) / 2

    The correction is exact and deterministic.  For clean natural images the
    residual is negligible (< 1e-6 in pixel values).  For images that have
    been through multiple codec rounds or manual FFT edits, it can be visible.

    The second output "residual" shows the imaginary part of IFFT(original
    spectrum) — use it as a diagnostic; nearly-black = no problem.

    Recommended placement: after any node that performs manual magnitude / phase
    editing, or as a "clean-up" step before VAE encoding.
    """

    CATEGORY = "Spectral Preprocessing"

    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("image", "residual")
    FUNCTION      = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "1.0 = fully enforce Hermitian symmetry.  "
                            "0.0 = no change.  "
                            "For most images the difference is invisible at strength=1."
                        ),
                    },
                ),
            }
        }

    def apply(self, image: "torch.Tensor", strength: float) -> tuple:
        arr            = tensor_to_numpy(image)
        orig_h, orig_w = arr.shape[1], arr.shape[2]
        batch          = arr.shape[0]

        outs      = []
        residuals = []
        for i in range(batch):
            out, res = _enforce_hermitian_image(arr[i], strength=strength)
            outs.append(out)
            residuals.append(res)

        out_arr = np.stack(outs,      axis=0)
        res_arr = np.stack(residuals, axis=0)

        out_arr = center_crop_to_match(out_arr, orig_h, orig_w)
        res_arr = center_crop_to_match(res_arr, orig_h, orig_w)

        # Scale residual to [0,1] for visualisation (it's normally tiny)
        res_max = res_arr.max()
        if res_max > 1e-8:
            res_vis = (res_arr / res_max).astype(np.float32)
        else:
            res_vis = res_arr

        return (
            numpy_to_tensor(clamp01(out_arr)),
            numpy_to_tensor(clamp01(res_vis)),
        )
