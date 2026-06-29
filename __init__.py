"""
Spectral Preprocessing Nodes for ComfyUI
=========================================
FFT-based image preprocessing nodes designed to reduce latent instability
before Flux image-to-image workflows.

These nodes do NOT optimise for perceptual image quality.  Their objective
is to suppress frequency-domain artifacts that are nearly invisible to humans
but destabilise the VAE encoder and the diffusion model.

Registered nodes
----------------
Artifact suppression:
  SpectralSpikeSuppressor        — attenuate isolated FFT magnitude spikes
  RadialSpectrumNormalizer       — nudge radial power spectrum toward 1/f^α
  AdaptiveHFCompressor           — soft-knee compression of HF band energy
  DirectionalArtifactSuppressor  — attenuate directional anomalies in FFT space
  LogSpectrumPeakCompressor      — homomorphic peak compression in log-mag domain
  NoiseFloorLifter               — spectral subtraction of the uniform noise floor

Colour correction:
  SpectralChannelEqualizer       — reduce inter-channel spectral mismatch

Frequency-domain blending:
  SpectralMagnitudeBlend         — blend FFT magnitudes, keep phase from A
  SpectralPhaseBlend             — blend FFT phases (circular), keep magnitude from A

Debug:
  FFTSpectrumVisualizer          — 6-panel spectral diagnostic grid
"""

from .nodes.spectral_spike_suppressor        import SpectralSpikeSuppressor
from .nodes.radial_spectrum_normalizer       import RadialSpectrumNormalizer
from .nodes.adaptive_hf_compressor          import AdaptiveHFCompressor
from .nodes.directional_artifact_suppressor  import DirectionalArtifactSuppressor
from .nodes.log_spectrum_peak_compressor    import LogSpectrumPeakCompressor
from .nodes.noise_floor_lifter              import NoiseFloorLifter
from .nodes.spectral_channel_equalizer      import SpectralChannelEqualizer
from .nodes.spectral_magnitude_blend        import SpectralMagnitudeBlend
from .nodes.spectral_phase_blend            import SpectralPhaseBlend
from .nodes.spectrum_visualizer             import FFTSpectrumVisualizer

NODE_CLASS_MAPPINGS = {
    "SpectralSpikeSuppressor":        SpectralSpikeSuppressor,
    "RadialSpectrumNormalizer":       RadialSpectrumNormalizer,
    "AdaptiveHFCompressor":           AdaptiveHFCompressor,
    "DirectionalArtifactSuppressor":  DirectionalArtifactSuppressor,
    "LogSpectrumPeakCompressor":      LogSpectrumPeakCompressor,
    "NoiseFloorLifter":               NoiseFloorLifter,
    "SpectralChannelEqualizer":       SpectralChannelEqualizer,
    "SpectralMagnitudeBlend":         SpectralMagnitudeBlend,
    "SpectralPhaseBlend":             SpectralPhaseBlend,
    "FFTSpectrumVisualizer":          FFTSpectrumVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectralSpikeSuppressor":        "Spectral Spike Suppressor",
    "RadialSpectrumNormalizer":       "Radial Spectrum Normalizer",
    "AdaptiveHFCompressor":           "Adaptive HF Compressor",
    "DirectionalArtifactSuppressor":  "Directional Artifact Suppressor",
    "LogSpectrumPeakCompressor":      "Log Spectrum Peak Compressor",
    "NoiseFloorLifter":               "Noise Floor Lifter",
    "SpectralChannelEqualizer":       "Spectral Channel Equalizer",
    "SpectralMagnitudeBlend":         "Spectral Magnitude Blend",
    "SpectralPhaseBlend":             "Spectral Phase Blend",
    "FFTSpectrumVisualizer":          "FFT Spectrum Visualizer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
