"""Helper for running remote deployment commands on the Linux host.

Reads SSH credentials from env vars or falls back to the project-specific
SSH key generated during D1. Never embed the password in this file.

Usage:
    # Run a shell command on the host
    python scripts/deploy_remote.py <cmd>

    # Upload a file (streams via `cat > remote` to avoid SFTP quirks)
    python scripts/deploy_remote.py --put <local> <remote>

    # Read a remote file
    python scripts/deploy_remote.py --cat <remote>

Environment variables:
    DEPLOY_HOST              override target host (default: 192.168.18.222)
    DEPLOY_USER              override remote user (default: marty)
    DEPLOY_HOST_PASSWORD     password fallback if SSH key auth fails
    DEPLOY_KEY_PATH          override SSH key (default: ~/.ssh/daytrader_deploy_id)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

HOST = os.environ.get("DEPLOY_HOST", "192.168.18.222")
USER = os.environ.get("DEPLOY_USER", "marty")
KEY_PATH = Path(os.environ.get("DEPLOY_KEY_PATH", str(Path.home() / ".ssh" / "daytrader_deploy_id")))


def _client() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Prefer key auth — D1 installed our pubkey on the host, so we
    # shouldn't need a password for any normal operation.
    if KEY_PATH.is_file():
        try:
            c.connect(
                HOST, username=USER,
                key_filename=str(KEY_PATH),
                timeout=10,
                look_for_keys=False, allow_agent=False,
            )
            return c
        except paramiko.AuthenticationException:
            pass  # key not authorized — fall through to password

    pw = os.environ.get("DEPLOY_HOST_PASSWORD")
    if not pw:
        sys.exit(
            f"SSH key auth to {USER}@{HOST} failed and DEPLOY_HOST_PASSWORD is unset. "
            "Either re-run D1 to install the key, or `export DEPLOY_HOST_PASSWORD=...`."
        )
    c.connect(
        HOST, username=USER, password=pw,
        timeout=10,
        look_for_keys=False, allow_agent=False,
    )
    return c


def run(cmd: str) -> int:
    with _client() as c:
        stdin, stdout, stderr = c.exec_command(cmd)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        status = stdout.channel.recv_exit_status()
    if out:
        sys.stdout.write(out + ("" if out.endswith("\n") else "\n"))
    if err:
        sys.stderr.write(f"[stderr] {err}" + ("" if err.endswith("\n") else "\n"))
    return status


def put(local: str, remote: str) -> None:
    """Upload by streaming bytes through `cat > remote`."""
    data = Path(local).resolve().read_bytes()
    with _client() as c:
        stdin, stdout, stderr = c.exec_command(f"cat > {remote!s}")
        stdin.write(data)
        stdin.channel.shutdown_write()
        status = stdout.channel.recv_exit_status()
    if status != 0:
        err = stderr.read().decode(errors="replace") if stderr else ""
        raise SystemExit(f"upload failed ({status}): {err}")
    sys.stdout.write(f"uploaded {local} -> {remote} ({len(data)} bytes)\n")


def main(argv: list[str]) -> int:
    if not argv:
        sys.stdout.write(__doc__ or "")
        return 2
    if argv[0] == "--put" and len(argv) == 3:
        put(argv[1], argv[2])
        return 0
    if argv[0] == "--cat" and len(argv) == 2:
        return run(f"cat {argv[1]}")
    return run(" ".join(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
