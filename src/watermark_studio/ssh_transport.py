"""Headless SSH transport selection shared by WSL clients and delivery workers."""
from pathlib import Path


def default_ssh_program():
    # Use the host SSH transport in WSL; model, GUI and dependencies remain Linux.
    native = Path("/mnt/c/Windows/System32/OpenSSH/ssh.exe")
    release = Path("/proc/sys/kernel/osrelease")
    if native.is_file() and release.exists() and "microsoft" in release.read_text().lower():
        return str(native)
    return "ssh"
