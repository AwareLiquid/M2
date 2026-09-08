import sys

from .research import hebbian_plasticity as _implementation

sys.modules[__name__] = _implementation
