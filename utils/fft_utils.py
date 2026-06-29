"""
Shared FFT utilities for Spectral Preprocessing Nodes.

All heavy lifting (tensor↔numpy, FFT/IFFT, batch iteration) lives here so
individual node files stay focused on their algorithm and nothing is duplicated.

Image convention (ComfyUI):
    torch.Tensor  shape [B, H, W, C]  float32  range [0, 1]  channels = RGB

NumPy convention used internally:
    np.ndarray    shape [H, W, C]     float32  range [0, 1]
"""

import numpy as np
import torch
from scipy.ndimage import median_filter, uniform_filter


# ---------------------------------------------------------------------------
# Tensor ↔ NumPy
# ---------------------------------------------------------------------------

def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    """[B, H, W, C] tensor → [B, H, W, C] float32 numpy array."""
    return image.cpu().numpy().astype(np.float32)


def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """[B, H, W, C] float32 numpy array → [B, H, W, C] tensor."""
    return torch.from_numpy(array)


# ---------------------------------------------------------------------------
# Per-channel FFT helpers
# ---------------------------------------------------------------------------

def fft2_channel(channel: np.ndarray) -> np.ndarray:
    """
    2-D FFT of a single [H, W] float32 channel.

    Returns the *shifted* complex spectrum (DC at center) as complex128.
    Shifting is done here so every caller works in the same coordinate system.
    """
    return np.fft.fftshift(np.fft.fft2(channel))


def ifft2_channel(spectrum: np.ndarray) -> np.ndarray:
    """
    Inverse of fft2_channel.  Accepts a shifted complex spectrum, returns a
    real [H, W] float32 image.  The imaginary residual after IFFT is discarded
    (it is always negligible for physically meaningful spectra).
    """
    return np.fft.ifft2(np.fft.ifftshift(spectrum)).real.astype(np.float32)


def magnitude(spectrum: np.ndarray) -> np.ndarray:
    """Magnitude of a complex spectrum array."""
    return np.abs(spectrum)


def phase(spectrum: np.ndarray) -> np.ndarray:
    """Phase angle of a complex spectrum array."""
    return np.angle(spectrum)


def reconstruct(mag: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """Reconstruct a complex spectrum from separate magnitude and phase."""
    return mag * np.exp(1j * ph)


# ---------------------------------------------------------------------------
# Radial coordinate grid
# ---------------------------------------------------------------------------

def radial_grid(h: int, w: int) -> np.ndarray:
    """
    Return a [H, W] array of radial distances from the DC centre.

    The maximum radius is normalised to 1.0 so the grid is independent of
    image resolution.
    """
    cy, cx = h // 2, w // 2
    y = np.arange(h, dtype=np.float32) - cy
    x = np.arange(w, dtype=np.float32) - cx
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx ** 2 + yy ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    return r / (max_r + 1e-8)


# ---------------------------------------------------------------------------
# Median-filter background estimation
# ---------------------------------------------------------------------------

def estimate_background_median(mag: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Estimate the smooth background of a magnitude spectrum by median-filtering.

    kernel_size should be odd; if even it will be incremented by 1.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return median_filter(mag, size=kernel_size)


def estimate_background_mean(mag: np.ndarray, kernel_size: int) -> np.ndarray:
    """Alternative background via uniform (mean) filter — faster, slightly less robust."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return uniform_filter(mag, size=kernel_size)


# ---------------------------------------------------------------------------
# Batch processing helper
# ---------------------------------------------------------------------------

def process_batch(images: np.ndarray, fn, **kwargs) -> np.ndarray:
    """
    Apply fn(image_hw_c, **kwargs) → image_hw_c to every item in a batch.

    images : [B, H, W, C]
    fn     : callable([H, W, C]) → [H, W, C]
    Returns: [B, H, W, C]
    """
    return np.stack([fn(images[b], **kwargs) for b in range(images.shape[0])], axis=0)


# ---------------------------------------------------------------------------
# DC protection helper
# ---------------------------------------------------------------------------

def get_dc_mask(h: int, w: int, radius: int = 2) -> np.ndarray:
    """
    Boolean mask that is True in a small square around DC (centre of the
    shifted spectrum).  Used by nodes that offer a preserve_dc option.

    radius : half-size of the protected square in pixels
    """
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2
    r = max(1, radius)
    mask[cy - r: cy + r + 1, cx - r: cx + r + 1] = True
    return mask


# ---------------------------------------------------------------------------
# Angular coordinate grid
# ---------------------------------------------------------------------------

def angular_grid(h: int, w: int) -> np.ndarray:
    """
    [H, W] array of angles in [0, π) measured from the positive x-axis,
    with the DC centre at (h//2, w//2).  Returned in radians.

    The Fourier magnitude has 180° periodicity (F(−r) = F*(r)), so angles
    are folded into [0, π) — callers that need the full [0, 2π) range should
    use np.arctan2 directly.
    """
    cy, cx = h // 2, w // 2
    y = np.arange(h, dtype=np.float32) - cy
    x = np.arange(w, dtype=np.float32) - cx
    xx, yy = np.meshgrid(x, y)
    return (np.arctan2(yy, xx) % np.pi).astype(np.float32)


# ---------------------------------------------------------------------------
# Numerically stable sigmoid
# ---------------------------------------------------------------------------

def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid that avoids exp overflow.

    For x >= 0: sigmoid = 1 / (1 + exp(-x))   — exp(-x) ≤ 1, safe.
    For x <  0: sigmoid = exp(x) / (1 + exp(x)) — exp(x) < 1, safe.

    Both branches produce identical values; the split prevents exp(large_positive).
    """
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[ pos] = 1.0 / (1.0 + np.exp(-x[ pos]))
    neg = ~pos
    ex  = np.exp(x[neg])
    out[neg]  = ex / (1.0 + ex)
    return out


# ---------------------------------------------------------------------------
# Tiled processing
# ---------------------------------------------------------------------------

def _hann2d(h: int, w: int) -> np.ndarray:
    """
    2-D Hann window [H, W] float32 for smooth tile blending.

    np.hanning(N) is exactly 0.0 at indices 0 and N-1.  If a tile lands at the
    image border those pixels get weight=0 and their processed value is dropped.
    Using np.hanning(N+2)[1:-1] gives the same cosine shape but non-zero at both
    endpoints (min value ≈ 1/(N+1)²), so every pixel always receives a contribution.
    """
    hy = np.hanning(h + 2)[1:-1].astype(np.float32)
    hx = np.hanning(w + 2)[1:-1].astype(np.float32)
    return np.outer(hy, hx)


def _tile_positions(length: int, tile: int, stride: int) -> list:
    """
    Return start positions for 1-D tiling so every pixel is covered.
    The last tile is snapped to the end of the array if needed.
    """
    positions = list(range(0, length - tile + 1, stride))
    if not positions or positions[-1] + tile < length:
        positions.append(max(0, length - tile))
    return positions


def process_image_tiled(image: np.ndarray, fn, tile_size: int, tile_overlap: int, **kwargs) -> np.ndarray:
    """
    Apply fn([H, W, C]) → [H, W, C] to a single image using overlapping tiles.

    Tiles are blended with a 2-D Hann window (overlap-add), so boundaries
    between tiles are invisible.  Falls back to whole-image processing when the
    image is smaller than tile_size in either dimension.

    Parameters
    ----------
    image        : [H, W, C] float32
    fn           : callable([H, W, C], **kwargs) → [H, W, C]
    tile_size    : size of each square tile in pixels
    tile_overlap : overlap between adjacent tiles in pixels (must be < tile_size)
    """
    h, w, c = image.shape

    if h <= tile_size and w <= tile_size:
        return fn(image, **kwargs)

    tile_overlap = min(tile_overlap, tile_size - 1)
    stride = max(1, tile_size - tile_overlap)

    out     = np.zeros((h, w, c), dtype=np.float32)
    weights = np.zeros((h, w),    dtype=np.float32)
    window  = _hann2d(tile_size, tile_size)

    for y0 in _tile_positions(h, tile_size, stride):
        for x0 in _tile_positions(w, tile_size, stride):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            th, tw = y1 - y0, x1 - x0

            tile      = image[y0:y1, x0:x1, :]
            processed = fn(tile, **kwargs)
            win2d     = window[:th, :tw]

            out[y0:y1, x0:x1, :]  += processed * win2d[:, :, np.newaxis]
            weights[y0:y1, x0:x1] += win2d

    weights = np.maximum(weights, 1e-8)
    return (out / weights[:, :, np.newaxis]).astype(np.float32)


def process_batch_tiled(images: np.ndarray, fn, tile_size: int, tile_overlap: int, **kwargs) -> np.ndarray:
    """
    Tiled variant of process_batch.  When tile_size <= 0, falls back to
    whole-image processing so callers can use this unconditionally.
    """
    if tile_size <= 0:
        return process_batch(images, fn, **kwargs)
    return np.stack(
        [process_image_tiled(images[b], fn, tile_size, tile_overlap, **kwargs)
         for b in range(images.shape[0])],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Phase interpolation (circular)
# ---------------------------------------------------------------------------

def lerp_phase(phase_a: np.ndarray, phase_b: np.ndarray, t: float) -> np.ndarray:
    """
    Circular interpolation between two phase arrays.

    Naive linear interpolation of angles fails at the ±π wrap boundary.
    This converts phases to unit complex vectors, interpolates in the complex
    plane, then extracts the angle — correctly handling the wrap.

    t=0 → phase_a,  t=1 → phase_b
    """
    a = np.exp(1j * phase_a)
    b = np.exp(1j * phase_b)
    mixed = (1.0 - t) * a + t * b
    # mixed may have near-zero magnitude at cancellation points; angle is
    # still well-defined (atan2(0,0) = 0) but we get a harmless 0 there.
    return np.angle(mixed).astype(np.float64)


# ---------------------------------------------------------------------------
# Image resize (nearest-neighbour, for mismatched blend inputs)
# ---------------------------------------------------------------------------

def resize_to_match(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize [H, W, C] image to (target_h, target_w) via nearest-neighbour.
    Used when two images fed to a blend node have different resolutions.
    """
    h, w, c = image.shape
    if h == target_h and w == target_w:
        return image
    ry = np.linspace(0, h - 1, target_h).astype(int)
    rx = np.linspace(0, w - 1, target_w).astype(int)
    return image[np.ix_(ry, rx)].copy()


# ---------------------------------------------------------------------------
# Output size guard (center-crop / pad to match input)
# ---------------------------------------------------------------------------

def center_crop_to_match(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Ensure a [B, H, W, C] array has exactly (target_h, target_w) spatial dims.

    If the processed array is larger than the target it is center-cropped.
    If it is smaller it is edge-padded.  When sizes already match this is a
    no-op (no copy, just returns arr).

    Called at the end of every node's apply() so that floating-point rounding
    or tiling boundary effects can never propagate a size mismatch to downstream
    nodes in the workflow.
    """
    _, h, w, _ = arr.shape
    if h == target_h and w == target_w:
        return arr

    # --- height ---
    if h > target_h:
        y0 = (h - target_h) // 2
        arr = arr[:, y0: y0 + target_h, :, :]
    elif h < target_h:
        pad = target_h - h
        arr = np.pad(arr, ((0, 0), (pad // 2, pad - pad // 2), (0, 0), (0, 0)), mode="edge")

    # --- width ---
    _, h, w, _ = arr.shape   # h is now correct; re-read w
    if w > target_w:
        x0 = (w - target_w) // 2
        arr = arr[:, :, x0: x0 + target_w, :]
    elif w < target_w:
        pad = target_w - w
        arr = np.pad(arr, ((0, 0), (0, 0), (pad // 2, pad - pad // 2), (0, 0)), mode="edge")

    return arr


# ---------------------------------------------------------------------------
# Safe clamp
# ---------------------------------------------------------------------------

def clamp01(array: np.ndarray) -> np.ndarray:
    """Clamp to [0, 1] and return float32."""
    return np.clip(array, 0.0, 1.0).astype(np.float32)
