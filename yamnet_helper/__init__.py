"""
YAMNet Helper Module
Provides tools for audio classification using YAMNet with magnitude spectra input.
"""

from .yamnet_spectrum_classifier import (
    YAMNetSpectrumClassifier,
    compute_magnitude_spectrum
)

__all__ = [
    'YAMNetSpectrumClassifier',
    'compute_magnitude_spectrum'
]

__version__ = '1.0.0'
