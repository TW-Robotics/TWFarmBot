"""Self-signed TLS for the dashboard so remote browsers get a secure context."""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path


def tls_dir() -> Path:
    return Path(os.getenv("TWFB_UI_TLS_DIR", "data/ui_tls"))


def local_ips() -> list[str]:
    ips = {"127.0.0.1"}
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


def ensure_certs(directory: Path | None = None) -> tuple[Path, Path]:
    folder = directory or tls_dir()
    folder.mkdir(parents=True, exist_ok=True)
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
