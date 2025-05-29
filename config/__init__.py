print('lcpvfxtools.config initialised')

# Other constants
SUPPORTED_RAW_FORMATS = ["cr2", "nef", "arw", "dng"]

# lcpvfxtools/config/__init__.py

try:
    from ._local_config import *  # user config (not committed)
except ImportError:
    from ._default import *       # fallback if no user config


import os
if not os.path.exists(__file__.replace('__init__.py', '_local_config.py')):
    import warnings
    warnings.warn("Using default config. Copy _default.py to _local_config.py and edit paths.")
else:
    print("Using local configuration settings from _local_config.py")