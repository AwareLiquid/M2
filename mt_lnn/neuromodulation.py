import sys

from .research import neuromodulation as _implementation

sys.modules[__name__] = _implementation
