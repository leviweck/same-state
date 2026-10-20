"""Suite-wide network guard: no test may reach anything but this machine.

The guard (tests/netguard/netguard.py) patches the socket layer, not one HTTP function: TCP connect/connect_ex,
UDP sendto/sendmsg, getaddrinfo, gethostbyname and gethostbyname_ex. Loopback stays allowed for local listeners.
Child Python processes are guarded too: tests/netguard is put first on PYTHONPATH with PLM_KIT_TEST_NETGUARD=1, and
its sitecustomize.py installs the same guard at interpreter start. Non-Python subprocesses are not covered by
this; the suite also passes under `unshare -rn sh -c 'ip link set lo up && pytest'`, which covers everything."""
import os
import pathlib
import sys

import pytest

GUARD_DIR = pathlib.Path(__file__).resolve().parent / "netguard"
sys.path.insert(0, str(GUARD_DIR))

import netguard

NetworkBlockedError = netguard.NetworkBlockedError


@pytest.fixture(autouse=True, scope="session")
def _guard_child_processes():
    mp = pytest.MonkeyPatch()
    mp.setenv("PLM_KIT_TEST_NETGUARD", "1")
    mp.setenv("PYTHONPATH", os.pathsep.join([str(GUARD_DIR), *filter(None, [os.environ.get("PYTHONPATH")])]))
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _block_network():
    # Own MonkeyPatch (not the test's `monkeypatch`), so a test's monkeypatch.undo() can't lift the guard.
    mp = pytest.MonkeyPatch()
    netguard.install(mp)
    yield
    mp.undo()
