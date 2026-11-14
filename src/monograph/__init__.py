"""monograph — diary -> memoir -> narrated audio, local-first.

Only light things are imported at top level so `import monograph` stays cheap and
free of heavy optional dependencies. Pull backends from their submodules
(`monograph.narrate`, `monograph.index`, ...) when you need them.
"""
from .config import Config
from . import types

__version__ = "0.1.0"
__all__ = ["Config", "types", "__version__"]
