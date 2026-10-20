"""Installs the test network guard in child Python processes started by the test suite.

tests/conftest.py puts this folder first on PYTHONPATH and sets PLM_KIT_TEST_NETGUARD=1 for the session, so any
`python ...` subprocess a test starts is guarded too. Any sitecustomize this one shadows is still loaded."""
import importlib
import os
import sys

if os.environ.get("PLM_KIT_TEST_NETGUARD") == "1":
    import netguard

    netguard.install()

_here = os.path.dirname(os.path.abspath(__file__))
_saved = sys.path[:]
try:                                                    # chain-load the sitecustomize we shadow (e.g. the distro's)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
    _this = sys.modules.pop("sitecustomize", None)
    try:
        importlib.import_module("sitecustomize")
    except ImportError:
        pass
    finally:
        if _this is not None:
            sys.modules["sitecustomize"] = _this
finally:
    sys.path[:] = _saved
