import sys

from .research import predictive_coding as _implementation

sys.modules[__name__] = _implementation
