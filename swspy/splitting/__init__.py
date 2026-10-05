"""
Submodule for undertaking splitting analysis.
"""

# Import neccessary modules:
from .split_windowcheck import *
# forward_model.py (synthetic forward-modeling of split waveforms) imports `split` (the
# original, unmodified swspy module we removed in favor of split_windowcheck) and isn't
# needed for measurement -- left out rather than porting an unused dependency.