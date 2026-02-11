#!/usr/bin/env python3
"""
Minimal example: Use YAMNet as a blackbox classifier.

This is the simplest possible usage - just copy this pattern!
"""

import numpy as np
from simple_demo import SimpleYAMNetClassifier

def main():
    # 1. Initialize classifier (once)
    print("Initializing YAMNet...")
    classifier = SimpleYAMNetClassifier()
    
    # 2. Generate or load your magnitude spectra
    #    Each spectrum should be 257 floats (output of 512-point FFT)
    print("\nProcessing spectra...")
    
    # Example: synthetic random spectra for demonstration
    for i in range(150):
        # Your spectrum here (257 bins)
        spectrum = np.random.rand(257).astype(np.float32) * 0.5
        
        # 3. Feed to classifier
        result = classifier.predict(spectrum)
        
        # 4. Check if prediction is ready (every 96 frames with overlap)
        if result:
            print(f"Frame {i:3d}: {result['class']:30s} confidence={result['confidence']:.3f}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
