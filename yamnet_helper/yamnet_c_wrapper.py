"""
Python wrapper for the C++ YAMNet implementation.
Uses ctypes to call the C API from z_odas_newbeamform.
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class YAMNetCWrapper:
    """
    Python wrapper for C++ YAMNet classifier.
    Calls the compiled C library directly for consistent behavior with ODAS.
    """
    
    def __init__(self, lib_path: str, model_path: str, class_map_path: str):
        """
        Initialize YAMNet classifier using C++ library.
        
        Args:
            lib_path: Path to the compiled shared library (e.g., libyamnet.so)
            model_path: Path to yamnet_core.tflite
            class_map_path: Path to yamnet_class_map.csv
        """
        # Load the shared library
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self._define_function_signatures()
        
        # Create classifier instance
        model_path_bytes = model_path.encode('utf-8')
        class_map_path_bytes = class_map_path.encode('utf-8')
        
        self.handle = self.lib.yamnet_create(model_path_bytes, class_map_path_bytes)
        
        if not self.handle:
            raise RuntimeError("Failed to create YAMNet classifier")
        
        # Get number of classes
        self.num_classes = self.lib.yamnet_num_classes(self.handle)
        
        print(f"YAMNet C++ Wrapper initialized:")
        print(f"  Model: {model_path}")
        print(f"  Classes: {self.num_classes}")
    
    def _define_function_signatures(self):
        """Define C function signatures for ctypes."""
        
        # yamnet_create(const char* model_path, const char* class_map_path) -> yamnet_handle_t*
        self.lib.yamnet_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.yamnet_create.restype = ctypes.c_void_p
        
        # yamnet_destroy(yamnet_handle_t* handle)
        self.lib.yamnet_destroy.argtypes = [ctypes.c_void_p]
        self.lib.yamnet_destroy.restype = None
        
        # yamnet_classify_patch(handle, patch, class_id, class_name, confidence) -> int
        self.lib.yamnet_classify_patch.argtypes = [
            ctypes.c_void_p,                    # handle
            ctypes.POINTER(ctypes.c_float),     # patch
            ctypes.POINTER(ctypes.c_int),       # out_class_id
            ctypes.POINTER(ctypes.c_char_p),    # out_class_name
            ctypes.POINTER(ctypes.c_float)      # out_confidence
        ]
        self.lib.yamnet_classify_patch.restype = ctypes.c_int
        
        # yamnet_add_frame(handle, spectrum, class_id, class_name, confidence) -> int
        self.lib.yamnet_add_frame.argtypes = [
            ctypes.c_void_p,                    # handle
            ctypes.POINTER(ctypes.c_float),     # spectrum
            ctypes.POINTER(ctypes.c_int),       # out_class_id
            ctypes.POINTER(ctypes.c_char_p),    # out_class_name
            ctypes.POINTER(ctypes.c_float)      # out_confidence
        ]
        self.lib.yamnet_add_frame.restype = ctypes.c_int
        
        # yamnet_reset(handle)
        self.lib.yamnet_reset.argtypes = [ctypes.c_void_p]
        self.lib.yamnet_reset.restype = None
        
        # yamnet_num_classes(handle) -> int
        self.lib.yamnet_num_classes.argtypes = [ctypes.c_void_p]
        self.lib.yamnet_num_classes.restype = ctypes.c_int
        
        # yamnet_class_name_from_id(handle, class_id) -> const char*
        self.lib.yamnet_class_name_from_id.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.yamnet_class_name_from_id.restype = ctypes.c_char_p
    
    def classify_patch(self, patch: np.ndarray) -> Tuple[int, str, float]:
        """
        Classify a full patch of 96 frames × 257 bins.
        
        Args:
            patch: Array of shape (96, 257) containing magnitude spectra
            
        Returns:
            (class_id, class_name, confidence)
        """
        assert patch.shape == (96, 257), f"Expected shape (96, 257), got {patch.shape}"
        
        # Flatten patch to contiguous C array
        patch_flat = patch.flatten().astype(np.float32)
        patch_ptr = patch_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Prepare output variables
        class_id = ctypes.c_int()
        class_name_ptr = ctypes.c_char_p()
        confidence = ctypes.c_float()
        
        # Call C function
        result = self.lib.yamnet_classify_patch(
            self.handle,
            patch_ptr,
            ctypes.byref(class_id),
            ctypes.byref(class_name_ptr),
            ctypes.byref(confidence)
        )
        
        if result != 1:
            raise RuntimeError("Classification failed")
        
        class_name = class_name_ptr.value.decode('utf-8') if class_name_ptr.value else "Unknown"
        
        return class_id.value, class_name, confidence.value
    
    def add_frame(self, spectrum: np.ndarray) -> Optional[Tuple[int, str, float]]:
        """
        Add one frame of spectrum and return classification when ready.
        
        Args:
            spectrum: Array of shape (257,) containing magnitude values
            
        Returns:
            (class_id, class_name, confidence) if classification ready, None otherwise
        """
        assert spectrum.shape == (257,), f"Expected shape (257,), got {spectrum.shape}"
        
        # Convert to C array
        spectrum_c = spectrum.astype(np.float32)
        spectrum_ptr = spectrum_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Prepare output variables
        class_id = ctypes.c_int()
        class_name_ptr = ctypes.c_char_p()
        confidence = ctypes.c_float()
        
        # Call C function
        result = self.lib.yamnet_add_frame(
            self.handle,
            spectrum_ptr,
            ctypes.byref(class_id),
            ctypes.byref(class_name_ptr),
            ctypes.byref(confidence)
        )
        
        if result == 1:
            class_name = class_name_ptr.value.decode('utf-8') if class_name_ptr.value else "Unknown"
            return class_id.value, class_name, confidence.value
        
        return None
    
    def reset(self):
        """Clear the frame buffer."""
        self.lib.yamnet_reset(self.handle)
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from ID."""
        result = self.lib.yamnet_class_name_from_id(self.handle, class_id)
        if result:
            return result.decode('utf-8')
        return "Unknown"
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'handle') and self.handle:
            self.lib.yamnet_destroy(self.handle)


def find_yamnet_library() -> Optional[str]:
    """
    Try to find the compiled YAMNet shared library.
    
    Returns:
        Path to library or None if not found
    """
    search_paths = [
        "/home/azureuser/z_odas_newbeamform/build/libyamnet.so",
        "/home/azureuser/z_odas_newbeamform/lib/libyamnet.so",
        "/usr/local/lib/libyamnet.so",
        "./libyamnet.so",
    ]
    
    for path in search_paths:
        if Path(path).exists():
            return path
    
    return None


# Convenience function to automatically use C++ or Python implementation
def create_yamnet_classifier(model_path: str, class_map_path: str, 
                            prefer_cpp: bool = True):
    """
    Create YAMNet classifier, preferring C++ implementation if available.
    
    Args:
        model_path: Path to yamnet_core.tflite
        class_map_path: Path to yamnet_class_map.csv
        prefer_cpp: If True, try C++ implementation first
        
    Returns:
        YAMNetCWrapper or YAMNetSpectrumClassifier instance
    """
    if prefer_cpp:
        lib_path = find_yamnet_library()
        if lib_path:
            try:
                print(f"Using C++ implementation: {lib_path}")
                return YAMNetCWrapper(lib_path, model_path, class_map_path)
            except Exception as e:
                print(f"Failed to load C++ implementation: {e}")
                print("Falling back to Python implementation...")
    
    print("Using Python implementation")
    from yamnet_spectrum_classifier import YAMNetSpectrumClassifier
    return YAMNetSpectrumClassifier(model_path, class_map_path)
