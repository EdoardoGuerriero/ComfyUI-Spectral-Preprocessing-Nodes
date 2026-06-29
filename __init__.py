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
Bias removal:
  MoisanDecomposition            — remove cross artifact via periodic+smooth decomposition
  SpectralAngularEqualizer       — equalize systematic directional energy imbalances
  HermitianSymmetryEnforcer      — restore F[k,l]=conj(F[-k,-l]) symmetry constraint
  SpectralWhitening              — data-driven radial power normalization / whitening

Artifact suppression:
  SpectralSpikeSuppressor        — attenuate isolated FFT magnitude spikes
  RadialSpectrumNormalizer       — nudge radial power spectrum toward 1/f^α
  AdaptiveHFCompressor           — soft-knee compression of HF band energy
  DirectionalArtifactSuppressor  — attenuate directional anomalies in FFT space
  LogSpectrumPeakCompressor      — homomorphic peak compression in log-mag domain
  NoiseFloorLifter               — spectral subtraction of the uniform noise floor

Colour correction:
  SpectralChannelEqualizer       — reduce inter-channel spectral mismatch

Spectral matching:
  SpectralHistogramMatch         — match global FFT magnitude distribution to a reference
  RadialStratifiedHistogramMatch — per-band CDF matching (frequency-domain CLAHE)

Frequency-domain blending:
  SpectralMagnitudeBlend         — blend FFT magnitudes, keep phase from A
  SpectralPhaseBlend             — blend FFT phases (circular), keep magnitude from A

Debug:
  FFTSpectrumVisualizer          — 6-panel spectral diagnostic grid
"""

from .nodes.moisan_decomposition                   import MoisanDecomposition
from .nodes.spectral_angular_equalizer             import SpectralAngularEqualizer
from .nodes.hermitian_symmetry_enforcer            import HermitianSymmetryEnforcer
from .nodes.spectral_whitening                     import SpectralWhitening
from .nodes.spectral_spike_suppressor              import SpectralSpikeSuppressor
from .nodes.radial_spectrum_normalizer             import RadialSpectrumNormalizer
from .nodes.adaptive_hf_compressor                import AdaptiveHFCompressor
from .nodes.directional_artifact_suppressor        import DirectionalArtifactSuppressor
from .nodes.log_spectrum_peak_compressor          import LogSpectrumPeakCompressor
from .nodes.noise_floor_lifter                    import NoiseFloorLifter
from .nodes.spectral_channel_equalizer            import SpectralChannelEqualizer
from .nodes.spectral_magnitude_blend              import SpectralMagnitudeBlend
from .nodes.spectral_phase_blend                  import SpectralPhaseBlend
from .nodes.spectral_histogram_match              import SpectralHistogramMatch
from .nodes.radial_stratified_histogram_match     import RadialStratifiedHistogramMatch
from .nodes.spectrum_visualizer                   import FFTSpectrumVisualizer

NODE_CLASS_MAPPINGS = {
    "MoisanDecomposition":               MoisanDecomposition,
    "SpectralAngularEqualizer":          SpectralAngularEqualizer,
    "HermitianSymmetryEnforcer":         HermitianSymmetryEnforcer,
    "SpectralWhitening":                 SpectralWhitening,
    "SpectralSpikeSuppressor":           SpectralSpikeSuppressor,
    "RadialSpectrumNormalizer":          RadialSpectrumNormalizer,
    "AdaptiveHFCompressor":              AdaptiveHFCompressor,
    "DirectionalArtifactSuppressor":     DirectionalArtifactSuppressor,
    "LogSpectrumPeakCompressor":         LogSpectrumPeakCompressor,
    "NoiseFloorLifter":                  NoiseFloorLifter,
    "SpectralChannelEqualizer":          SpectralChannelEqualizer,
    "SpectralMagnitudeBlend":            SpectralMagnitudeBlend,
    "SpectralPhaseBlend":                SpectralPhaseBlend,
    "SpectralHistogramMatch":            SpectralHistogramMatch,
    "RadialStratifiedHistogramMatch":    RadialStratifiedHistogramMatch,
    "FFTSpectrumVisualizer":             FFTSpectrumVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoisanDecomposition":               "Moisan Decomposition",
    "SpectralAngularEqualizer":          "Spectral Angular Equalizer",
    "HermitianSymmetryEnforcer":         "Hermitian Symmetry Enforcer",
    "SpectralWhitening":                 "Spectral Whitening",
    "SpectralSpikeSuppressor":           "Spectral Spike Suppressor",
    "RadialSpectrumNormalizer":          "Radial Spectrum Normalizer",
    "AdaptiveHFCompressor":              "Adaptive HF Compressor",
    "DirectionalArtifactSuppressor":     "Directional Artifact Suppressor",
    "LogSpectrumPeakCompressor":         "Log Spectrum Peak Compressor",
    "NoiseFloorLifter":                  "Noise Floor Lifter",
    "SpectralChannelEqualizer":          "Spectral Channel Equalizer",
    "SpectralMagnitudeBlend":            "Spectral Magnitude Blend",
    "SpectralPhaseBlend":                "Spectral Phase Blend",
    "SpectralHistogramMatch":            "Spectral Histogram Match",
    "RadialStratifiedHistogramMatch":    "Radial Stratified Histogram Match",
    "FFTSpectrumVisualizer":             "FFT Spectrum Visualizer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
