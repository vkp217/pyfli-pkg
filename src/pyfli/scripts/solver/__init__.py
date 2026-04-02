##### inside solver.__init__.py
from .base_fitter import BaseFLIFitter
from .flicpuFitter import Fli_CPUProcessor
from .fligpuFitter import Fli_GPUProcessor
from .mleFitter import MLEFLIFitter
from .globalFitter import GlobalFLIFitter
from .comparison import FittingComparator
from .binned_fliFitter import BinnedFliFitter

# [BaseFLIFitter, Fli_CPUProcessor, Fli_GPUProcessor, MLEFLIFitter, GlobalFLIFitter, FittingComparator,
# BinnedFliFitter]