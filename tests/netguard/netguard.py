"""Network guard used by the test suite (tests/conftest.py) and by child Python processes (sitecustomize.py here).

Patched at the socket layer so every client is covered: urllib (urlopen or a custom OpenerDirector), requests,
raw TCP sockets, UDP datagrams and DNS lookups. Loopback (127.0.0.0/8, ::1, localhost) stays allowed.
Non-Python subprocesses (curl, ollama, ...) are NOT covered; run the suite under `unshare -rn` for that."""
import ipaddress
import socket


class NetworkBlockedError(AssertionError):
    """Raised when a test tries to reach anything but this machine."""


def is_loopback(host):
    if host in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


def is_ip_literal(host):
    try:
        ipaddress.ip_address(str(host).split("%")[0])
        return True
    except ValueError:
        return False


REAL = {
    "connect": socket.socket.connect,
    "connect_ex": socket.socket.connect_ex,
    "sendto": socket.socket.sendto,
    "sendmsg": socket.socket.sendmsg,
    "getaddrinfo": socket.getaddrinfo,
    "gethostbyname": socket.gethostbyname,
    "gethostbyname_ex": socket.gethostbyname_ex,
}


def _check_addr(addr, what):
    if isinstance(addr, tuple) and addr and not is_loopback(addr[0]):
        raise NetworkBlockedError(f"network blocked in tests: {what} to {addr[0]!r}")


def _check_name(host, what):
    if host is not None and not is_loopback(host) and not is_ip_literal(host):
        raise NetworkBlockedError(f"network blocked in tests: {what} of {host!r}")


def connect(self, addr):
    _check_addr(addr, "connect")
    return REAL["connect"](self, addr)


def connect_ex(self, addr):
    _check_addr(addr, "connect")
    return REAL["connect_ex"](self, addr)


def sendto(self, data, *args):
    _check_addr(args[-1] if args else None, "sendto")           # sendto(data, addr) / sendto(data, flags, addr)
    return REAL["sendto"](self, data, *args)


def sendmsg(self, buffers, *args):
    if len(args) >= 3:                                           # sendmsg(buffers, ancdata, flags, address)
        _check_addr(args[2], "sendmsg")
    return REAL["sendmsg"](self, buffers, *args)


def getaddrinfo(host, *a, **k):
    _check_name(host, "DNS lookup")
    return REAL["getaddrinfo"](host, *a, **k)


def gethostbyname(host):
    _check_name(host, "DNS lookup (gethostbyname)")
    return REAL["gethostbyname"](host)


def gethostbyname_ex(host):
    _check_name(host, "DNS lookup (gethostbyname_ex)")
    return REAL["gethostbyname_ex"](host)


PATCHES = [(socket.socket, "connect", connect), (socket.socket, "connect_ex", connect_ex),
           (socket.socket, "sendto", sendto), (socket.socket, "sendmsg", sendmsg),
           (socket, "getaddrinfo", getaddrinfo), (socket, "gethostbyname", gethostbyname),
           (socket, "gethostbyname_ex", gethostbyname_ex)]


def install(mp=None):
    """With a pytest MonkeyPatch: reversible. Without one (child processes): permanent for that process."""
    for obj, name, fn in PATCHES:
        if mp is not None:
            mp.setattr(obj, name, fn)
        else:
            setattr(obj, name, fn)
