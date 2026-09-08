import sys

from .research import global_coherence as _implementation

sys.modules[__name__] = _implementation
