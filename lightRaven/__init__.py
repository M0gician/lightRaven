import os
import sys
import platform
import textwrap
import ctypes

if sys.version_info < (3,):
    raise Exception("Python 2 has reached end-of-life and is no longer supported by lightRaven.")