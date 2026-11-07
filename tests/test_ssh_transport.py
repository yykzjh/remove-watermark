import subprocess

import pytest

from watermark_studio import production_local, ssh_transport


@pytest.mark.parametrize("kernel,present,host", [
    ("6.8-microsoft-standard-WSL2", True, True),
    ("6.8-generic", True, False),
    ("6.8-microsoft-standard-WSL2", False, False),
])
def test_host_ssh_is_selected_only_inside_wsl_when_available(tmp_path, monkeypatch, kernel, present, host):
    native, release = tmp_path / "host-ssh.exe", tmp_path / "osrelease"
    if present:
        native.touch()
    release.write_text(kernel)
    monkeypatch.setattr(ssh_transport, "Path",
                        lambda p: native if str(p).endswith("ssh.exe") else release)
    assert ssh_transport.default_ssh_program() == (str(native) if host else "ssh")


def test_headless_snapshot_uses_shared_transport_and_ascii_command(tmp_path, monkeypatch):
    monkeypatch.setattr(production_local, "default_ssh_program", lambda: "host-ssh.exe")
    commands = []
    def run(cmd, **kw):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")
    monkeypatch.setattr(production_local.subprocess, "run", run)
    assert production_local.remote_snapshot("cdu_dev2", str(tmp_path / "中文状态")) == {}
    assert commands[0][0] == "host-ssh.exe"
    assert "base64.b64decode" in commands[0][-1]
    commands[0][-1].encode("ascii")


def test_ssh_error_keeps_actionable_short_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setattr(production_local, "default_ssh_program", lambda: "ssh")
    monkeypatch.setattr(production_local.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(
                            cmd, 255, stdout="", stderr="Timeout, server not responding.\n"))
    with pytest.raises(ConnectionError, match="exit 255: Timeout"):
        production_local.remote_snapshot("cdu_dev2", str(tmp_path))
