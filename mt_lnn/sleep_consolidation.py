import sys

from .research import sleep_consolidation as _implementation

sys.modules[__name__] = _implementation
