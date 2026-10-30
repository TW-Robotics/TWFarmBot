"""TLS for the dashboard so remote browsers get a secure context.

Prefer a publicly trusted `tailscale cert` (Let's Encrypt) for this host's
MagicDNS name -- a self-signed cert always triggers Safari's "This Connection
Is Not Private" warning, no matter how correct its SANs are. Fall back to
self-signed (valid only via localhost/LAN after a manual trust warning).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
from pathlib import Path

_LOG = logging.getLogger(__name__)


def tls_dir() -> Path:
    return Path(os.getenv("TWFB_UI_TLS_DIR", "data/ui_tls"))


def local_ips() -> list[str]:
    ips = {"127.0.0.1"}
    # Enumerate every interface (incl. tailscale0's 100.x, which neither the
    # hostname nor the default-route probe below ever sees). Linux ioctl,
    # stdlib only; ponytail: non-Linux falls through to the probes below.
    try:
        import fcntl
        import struct

        for _, name in socket.if_nameindex():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                    packed = struct.pack("256s", name.encode()[:15])
                    ips.add(socket.inet_ntoa(fcntl.ioctl(probe.fileno(), 0x8915, packed)[20:24]))
            except OSError:
                continue
    except (ImportError, OSError):
        pass
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("1.1.1.1", 80))
        ips.add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127.") or ip == "127.0.0.1")


def san_entries() -> list[str]:
    names = ["DNS:localhost"]
    host = socket.gethostname().strip()
    if host and all(ch.isalnum() or ch in ".-" for ch in host):
        names.append(f"DNS:{host}")
    names.extend(f"IP:{ip}" for ip in local_ips())
    return names


def _tailscale_certs(folder: Path) -> tuple[Path, Path] | None:
    """Trusted cert for the MagicDNS name via `tailscale cert`, or None."""
    if not shutil.which("tailscale"):
        return None
    try:
        out = subprocess.run(
            ["tailscale", "status", "--json"], capture_output=True, text=True, timeout=10
        )
        domain = json.loads(out.stdout)["Self"]["DNSName"].rstrip(".")
        cert, key = folder / "tailscale.crt", folder / "tailscale.key"
        # --min-validity reuses the live cert, renews only when near expiry.
        subprocess.run(
            [
                "tailscale", "cert", "--min-validity=720h",
                f"--cert-file={cert}", f"--key-file={key}", domain,
            ],
            check=True, capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:  # noqa: BLE001 - any failure -> self-signed fallback
        _LOG.warning(
            "tailscale cert unavailable (%s); using self-signed cert. "
            "Fix: sudo tailscale set --operator=$USER, enable HTTPS in the "
            "Tailscale admin console, then restart the UI.", exc,
        )
        return None
    _LOG.info("UI TLS: trusted Tailscale cert; open https://%s", domain)
    return cert, key


def ensure_certs(directory: Path | None = None) -> tuple[Path, Path]:
    folder = directory or tls_dir()
    folder.mkdir(parents=True, exist_ok=True)
    trusted = _tailscale_certs(folder)
    if trusted is not None:
        return trusted
    cert = folder / "ui.crt"
    key = folder / "ui.key"
    stamp = folder / "san.txt"
    wanted = "\n".join(san_entries()) + "\n"
    if cert.is_file() and key.is_file() and stamp.read_text() == wanted:
        return cert, key
    san = ",".join(san_entries())
    try:
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-sha256",
                "-days",
                "825",
                "-nodes",
                "-keyout",
                str(key),
                "-out",
                str(cert),
                "-subj",
                "/CN=TWFarmBot",
                "-addext",
                f"subjectAltName={san}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(f"Could not create UI TLS certificate: {detail}") from exc
    stamp.write_text(wanted)
    return cert, key
