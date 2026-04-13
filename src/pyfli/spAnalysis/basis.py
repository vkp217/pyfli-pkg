import numpy as np
from scipy.fftpack import fwht, dct

class OrthogonalBasis:
    """Abstract class for sensing bases."""
    def forward(self, x): raise NotImplementedError
    def inverse(self, y): raise NotImplementedError

class HadamardBasis(OrthogonalBasis):
    def forward(self, x):
        # Image to Measurements
        return fwht(x.flatten(), axis=0)
    
    def inverse(self, y):
        # Measurements to Image
        return fwht(y, axis=0)

class DCTBasis(OrthogonalBasis):
    def forward(self, x):
        return dct(x.flatten(), norm='ortho')
    
    def inverse(self, y):
        return dct(y, type=3, norm='ortho')