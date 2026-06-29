"""
Node 13 — Moisan Periodic+Smooth Decomposition
================================================
Removes the cross artifact — the bright horizontal and vertical lines through
DC that appear in the FFT of virtually every natural image.

The cross artifact
------------------
FFT assumes its input is one period of an infinite periodic signal.  When the
pixel values at the top and bottom edges of an image differ (or left vs right),
the implicit periodic extension creates a hard "wrap discontinuity".  In the
Fourier domain this discontinuity shows up as concentrated energy along the
horizontal and vertical axes through DC — the characteristic "cross" visible
in the spectrum of almost any photograph.

This is not a noise artifact or a camera defect.  It is a mathematical
consequence of applying DFT to a non-periodic signal.  Every FFT-based node
in this pack (and in any other) processes this spurious cross energy along
with genuine image content.

Moisan's solution (2011)
------------------------
Lionel Moisan proved that any image I decomposes uniquely as:

    I = u + v

where:
    u  is the "periodic component"  — smoothly periodic, same mean as I,
                                      NO cross artifact in its FFT.
    v  is the "smooth component"    — a very smooth, nearly-flat field that
                                      exactly captures the boundary discontinuity.

The decomposition is computed analytically in the Fourier domain in a single
forward FFT + division + inverse FFT — no iterations, no parameters, exact.

Algorithm
---------
1.  Build the boundary image b from I:
      b[0,  :] = I[-1, :] − I[0,  :]   (top row:    bottom-of-image minus top)
      b[-1, :] = I[0,  :] − I[-1, :]   (bottom row: −b[0,:])
      b[:, 0 ] += I[:, -1] − I[:, 0 ]  (left col:   right-of-image minus left)
      b[:, -1] += I[:, 0 ] − I[:, -1]  (right col:  −b[:,0])
    b has exactly zero sum (sum(b) = 0) by construction.

2.  Compute V = FFT(b) / (2·cos(2πk/H) + 2·cos(2πl/W) − 4)
    Set V[0,0] = 0  (smooth component has the same mean as I).

3.  v = IFFT(V).real

4.  u = I − v

The node outputs u (the periodic, cross-free component) and optionally v.

Practical impact
----------------
Running any upstream FFT node (SpectralSpikeSuppressor, RadialNormalizer, etc.)
on u rather than I gives the same result without the cross artifact interfering.
After all FFT processing, u can be recombined with the smooth component v to
restore the original boundary behaviour:

    preprocessed = FFTNodes(u) + v

This node provides a `return_smooth` toggle to output v alongside u for exactly
this purpose, and a `strength` parameter to blend between I (0) and u (1) when
full debiasing is too aggressive.

Reference
---------
Moisan, L. (2011). Periodic Plus Smooth Image Decomposition.
Journal of Mathematical Imaging and Vision, 39(2), 161–179.
"""

import numpy as np

from ..utils.fft_utils import (
    tensor_to_numpy,
    numpy_to_tensor,
    center_crop_to_match,
    clamp01,
)


# ---------------------------------------------------------------------------
# Core Moisan algorithm (single channel)
# ---------------------------------------------------------------------------

def _moisan_channel(channel: np.ndarray) -> tuple:
    """
    Decompose a single [H, W] float32 channel into (u, v).

    u: periodic component (no cross artifact)
    v: smooth component   (captures boundary discontinuity)
    u + v = channel exactly.
    """
    H, W = channel.shape
    I = channel.astype(np.float64)

    # Step 1 — boundary image
    b = np.zeros((H, W), dtype=np.float64)
    b[0,  :] += I[-1, :] - I[0,  :]
    b[-1, :] += I[0,  :] - I[-1, :]
    b[:,  0] += I[:, -1] - I[:,  0]
    b[:, -1] += I[:,  0] - I[:, -1]

    # Step 2 — solve Poisson equation in Fourier domain
    B = np.fft.fft2(b)

    # Frequency coordinates (NOT shifted — Moisan works in standard FFT order)
    ky = (2.0 * np.pi / H) * np.arange(H, dtype=np.float64)
    kx = (2.0 * np.pi / W) * np.arange(W, dtype=np.float64)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")

    denom = 2.0 * np.cos(KY) + 2.0 * np.cos(KX) - 4.0
    denom[0, 0] = 1.0  # avoid division by zero; overwritten below

    V = B / denom
    V[0, 0] = 0.0  # smooth component has zero mean (same mean as I from u)

    # Step 3 — inverse FFT to get smooth component
    v = np.fft.ifft2(V).real.astype(np.float32)

    # Step 4 — periodic component
    u = (I - v).astype(np.float32)

    return u, v


# ---------------------------------------------------------------------------
# Per-image wrapper
# ---------------------------------------------------------------------------

def _moisan_image(image: np.ndarray, strength: float) -> tuple:
    """
    Apply Moisan decomposition to a [H, W, C] image.

    Returns (out, smooth) where:
        out   = I − strength · v  =  u  at strength=1, I at strength=0
        smooth = v  (the boundary smooth field, for optional recombination)
    """
    H, W, C = image.shape
    out    = np.empty_like(image)
    smooth = np.empty_like(image)

    for ch in range(C):
        u, v = _moisan_channel(image[:, :, ch])
        smooth[:, :, ch] = v
        # blend: out = I - strength * v = (1-strength)*I + strength*u
        out[:, :, ch] = image[:, :, ch] - strength * v

    return out, smooth


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class MoisanDecomposition:
    """
    Removes the FFT cross artifact via Moisan's Periodic+Smooth Decomposition.

    The "cross" (bright horizontal + vertical lines through DC in any FFT
    spectrum) is caused by the non-periodicity of the image at its borders.
    Moisan's algorithm separates an image into:

      • u  (periodic component) — spectrally clean, no cross artifact.
      • v  (smooth component)   — the boundary discontinuity field.

    The main output is u (strength=1) or a blend toward the original image
    (strength < 1).  The smooth component v is available as a second output
    for optional recombination after downstream spectral processing:

        result = SpectralNodes(u) + v

    This node has no trainable parameters — the decomposition is exact and
    deterministic.  The only free parameter is strength, which controls how
    aggressively the smooth component is subtracted.

    Recommended placement: first node in any FFT preprocessing chain.
    """

    CATEGORY = "Spectral Preprocessing"

    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("periodic_u", "smooth_v")
    FUNCTION      = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Input image to decompose."},
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "1.0 = output pure periodic component u (fully removes cross artifact).  "
                            "0.0 = output original image unchanged.  "
                            "Values between 0 and 1 partially subtract the smooth component."
                        ),
                    },
                ),
            }
        }

    def apply(self, image: "torch.Tensor", strength: float) -> tuple:
        arr            = tensor_to_numpy(image)
        orig_h, orig_w = arr.shape[1], arr.shape[2]
        batch          = arr.shape[0]

        outs    = []
        smooths = []
        for i in range(batch):
            u_img, v_img = _moisan_image(arr[i], strength=strength)
            outs.append(u_img)
            smooths.append(v_img)

        out_arr    = np.stack(outs,    axis=0)
        smooth_arr = np.stack(smooths, axis=0)

        out_arr    = center_crop_to_match(out_arr,    orig_h, orig_w)
        smooth_arr = center_crop_to_match(smooth_arr, orig_h, orig_w)

        # v can be negative (it's a correction field) — clamp u to [0,1] only;
        # output v as-is shifted to [0,1] for visualisation (it's nearly flat)
        v_vis = (smooth_arr - smooth_arr.min()) / (smooth_arr.ptp() + 1e-8)

        return (
            numpy_to_tensor(clamp01(out_arr)),
            numpy_to_tensor(v_vis.astype(np.float32)),
        )
