"""Outbound URL safety checks (SSRF hardening).

Run before the server makes a request to a *user-supplied* URL — e.g. the custom
embedding endpoint set via ``POST /api/embeddings/endpoint``, which then triggers
an outbound ``httpx`` call.

Odysseus is local-first: pointing the embedding endpoint at a loopback or LAN
address (a local vLLM / llama.cpp / Ollama server) is a normal, intended setup.
So this guard does **not** blanket-block private addresses by default — that would
break the primary use case. What it *always* rejects:

  - a non-HTTP(S) scheme (``file://``, ``gopher://``, ``ftp://`` …), and
  - the link-local range (``169.254.0.0/16`` / ``fe80::/10``), i.e. the cloud
    instance-metadata SSRF credential-exfil vector — nobody serves embeddings
    there — plus multicast / reserved / unspecified addresses.

For exposed multi-tenant deployments, set ``EMBEDDING_BLOCK_PRIVATE_IPS=true`` to
additionally reject all private and loopback targets (full SSRF lockdown).
"""

import ipaddress
import socket
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse

ALLOWED_SCHEMES = ("http", "https")

# RFC 6598 shared address space (carrier-grade NAT). It is not globally
# routable, but CPython does not classify it as ``is_private`` (it is "shared",
# not "private"), so the is_private/is_loopback checks miss it. Reject the range
# explicitly. This closes exactly the shared-space gap without coupling strict
# mode to ``is_global``'s broader definition, which has shifted across CPython
# versions for other special ranges.
_SHARED_ADDRESS_SPACE_V4 = ipaddress.ip_network("100.64.0.0/10")

# Well-known cloud instance metadata endpoints that do not sit in link-local (e.g. Alibaba Cloud 100.100.100.200).
_METADATA_IPS = {ipaddress.ip_address("100.100.100.200")}


class SSRFBlockedError(OSError):
    """Raised when an outbound connection attempt is blocked by SSRF policy."""
    pass


def _default_resolver(host: str) -> List[str]:
    """Resolve a hostname to the list of IP strings it maps to (A + AAAA)."""
    return [info[4][0] for info in socket.getaddrinfo(host, None)]


def _classify(ip: ipaddress._BaseAddress, *, block_private: bool) -> Optional[str]:
    """Return a rejection reason for an IP, or None if it is allowed."""
    # IPv4-mapped IPv6 (e.g. ::ffff:169.254.169.254) — judge the embedded v4.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_link_local or ip in _METADATA_IPS:
        return f"link-local address blocked (SSRF metadata risk): {ip}"
    if ip.is_loopback:
        if block_private:
            return f"private/shared/loopback address blocked: {ip}"
        return None
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return f"disallowed address: {ip}"
    if block_private and (
        ip.is_private
        or (isinstance(ip, ipaddress.IPv4Address) and ip in _SHARED_ADDRESS_SPACE_V4)
    ):
        return f"private/shared/loopback address blocked: {ip}"
    return None


def check_outbound_url(
    url: str,
    *,
    block_private: bool = False,
    resolver: Optional[Callable[[str], List[str]]] = None,
) -> Tuple[bool, str]:
    """Validate a user-supplied outbound URL.

    Returns ``(ok, reason)``. ``ok`` is True only when the URL is safe to fetch.
    ``resolver`` is injectable so callers/tests can avoid real DNS.
    """
    if not isinstance(url, str):
        return False, "URL must be a string"
    if not url or not url.strip():
        return False, "URL is required"
    try:
        parsed = urlparse(url.strip())
    except Exception as e:  # pragma: no cover - urlparse is very tolerant
        return False, f"unparseable URL: {e}"

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False, f"scheme must be http or https, got '{parsed.scheme or '(none)'}'"
    host = parsed.hostname
    if not host:
        return False, "URL has no host"

    resolve = resolver or _default_resolver
    try:
        raw_ips = resolve(host)
    except Exception as e:
        return False, f"host does not resolve: {e}"
    if not raw_ips:
        return False, "host does not resolve"

    saw_ip = False
    for raw in raw_ips:
        if not isinstance(raw, str):
            continue
        try:
            ip = ipaddress.ip_address(raw.split("%")[0])  # strip IPv6 zone id
        except ValueError:
            continue
        saw_ip = True
        reason = _classify(ip, block_private=block_private)
        if reason:
            return False, reason
    if not saw_ip:
        return False, "host does not resolve to an IP"
    return True, "ok"


_LOCALHOST_NAMES = ("localhost", "ip6-localhost", "ip6-loopback")


def _is_localhost_name(host: str) -> bool:
    h = host.lower().rstrip(".")
    return h in _LOCALHOST_NAMES or h.endswith(".localhost")


def check_outbound_host(
    host: str,
    *,
    block_private: bool = False,
    resolver: Optional[Callable[[str], List[str]]] = None,
    unresolved: str = "reject",
) -> Tuple[bool, str]:
    """Validate a bare user-supplied hostname / IP literal (no scheme).

    Companion to :func:`check_outbound_url` for protocols that take a host
    and port rather than a URL — IMAP, SMTP, CalDAV/CardDAV hosts, database
    DSNs. Same policy: link-local / multicast / reserved / unspecified
    addresses are always rejected (cloud metadata SSRF), private / loopback
    only when ``block_private`` is set. ``localhost`` spellings are treated as
    loopback without a DNS round-trip.

    ``unresolved`` controls what happens when a hostname does not resolve:

    - ``"reject"`` (default, fail-closed like ``check_outbound_url``);
    - ``"allow"``: pass it through. Use this only where the very next step
      is a connect that will surface the same DNS failure to the user, so
      an offline host or a flaky resolver does not turn into a spurious
      "blocked" error. IP literals and localhost names are still classified
      without DNS, so the metadata / loopback cases are covered either way.
    """
    if not isinstance(host, str):
        return False, "host must be a string"
    host = host.strip()
    if not host:
        return False, "host is required"
    # A bare host never carries a scheme, path, userinfo, port or whitespace.
    # Anything of the sort is either a copy/paste mistake or an attempt to
    # smuggle a target past the check; reject rather than guess.
    if any(ch.isspace() for ch in host) or any(ch in host for ch in "/\\@?#"):
        return False, "host must be a bare hostname or IP address"
    literal = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    if ":" in literal:
        # Only an IPv6 literal may contain colons; "host:port" is rejected.
        try:
            ip = ipaddress.ip_address(literal.split("%")[0])
        except ValueError:
            return False, "host must be a bare hostname or IP address"
        reason = _classify(ip, block_private=block_private)
        return (False, reason) if reason else (True, "ok")
    try:
        ip = ipaddress.ip_address(literal)
    except ValueError:
        ip = None
    if ip is not None:
        reason = _classify(ip, block_private=block_private)
        return (False, reason) if reason else (True, "ok")
    if _is_localhost_name(host):
        if block_private:
            return False, f"private/shared/loopback address blocked: {host}"
        return True, "ok"

    resolve = resolver or _default_resolver
    try:
        raw_ips = resolve(host)
    except Exception as e:
        if unresolved == "allow":
            return True, "unresolved (deferred to connect)"
        return False, f"host does not resolve: {e}"
    saw_ip = False
    for raw in raw_ips or []:
        if not isinstance(raw, str):
            continue
        try:
            resolved = ipaddress.ip_address(raw.split("%")[0])
        except ValueError:
            continue
        saw_ip = True
        reason = _classify(resolved, block_private=block_private)
        if reason:
            return False, reason
    if not saw_ip:
        if unresolved == "allow":
            return True, "unresolved (deferred to connect)"
        return False, "host does not resolve to an IP"
    return True, "ok"


def safe_create_connection(
    address: Tuple[str, int],
    timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
    source_address: Optional[Tuple[str, int]] = None,
    *,
    block_private: bool = True,
    resolver: Optional[Callable[[str], List[str]]] = None,
) -> socket.socket:
    """Establish a TCP connection guaranteed not to connect to prohibited IP addresses.

    Guards against DNS rebinding (TOCTOU) by resolving the hostname once, validating
    EVERY returned address against the SSRF policy, and connecting directly to the
    validated IP address. The caller retains the original hostname for TLS SNI and
    certificate verification.
    """
    host, port = address
    port = int(port or 0)
    if not isinstance(host, str):
        raise ValueError("host must be a string")
    host = host.strip()
    if not host:
        raise ValueError("host is required")
    if any(ch.isspace() for ch in host) or any(ch in host for ch in "/\\@?#"):
        raise ValueError("host must be a bare hostname or IP address")

    literal = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    ip = None
    if ":" in literal:
        try:
            ip = ipaddress.ip_address(literal.split("%")[0])
        except ValueError:
            pass
    else:
        try:
            ip = ipaddress.ip_address(literal)
        except ValueError:
            pass

    if ip is not None:
        reason = _classify(ip, block_private=block_private)
        if reason:
            raise SSRFBlockedError(f"Rejected mail host: {reason}")
        family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT and timeout is not None:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect((str(ip), port))
            return sock
        except Exception:
            sock.close()
            raise

    if _is_localhost_name(host):
        if block_private:
            raise SSRFBlockedError(f"Rejected mail host: private/shared/loopback address blocked: {host}")

    if resolver is not None:
        raw_ips = resolver(host)
        addr_list = []
        for raw in raw_ips or []:
            try:
                raw_ip = ipaddress.ip_address(raw.split("%")[0])
                af = socket.AF_INET6 if raw_ip.version == 6 else socket.AF_INET
                addr_list.append((af, socket.SOCK_STREAM, 6, "", (str(raw_ip), port)))
            except ValueError:
                continue
    else:
        try:
            addr_list = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
        except Exception as e:
            raise OSError(f"host does not resolve: {e}")

    if not addr_list:
        raise OSError("host does not resolve to an IP")

    validated_entries = []
    for res in addr_list:
        af, socktype, proto, canonname, sa = res
        ip_str = sa[0].split("%")[0]
        try:
            resolved_ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        reason = _classify(resolved_ip, block_private=block_private)
        if reason:
            raise SSRFBlockedError(f"Rejected mail host: {reason}")
        validated_entries.append(res)

    if not validated_entries:
        raise OSError("host does not resolve to a valid IP")

    exceptions = []
    for af, socktype, proto, canonname, sa in validated_entries:
        sock = None
        try:
            sock = socket.socket(af, socktype, proto)
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT and timeout is not None:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sa)
            return sock
        except OSError as exc:
            exceptions.append(exc)
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    if exceptions:
        raise exceptions[0]
    raise OSError(f"Failed to connect to {host}:{port}")
