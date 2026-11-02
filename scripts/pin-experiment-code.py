"""Create a fixed Git checkout from code already synchronized by Git; share Linux uv env."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def git(*args,cwd=None):
    return subprocess.check_output(["git",*args],cwd=cwd,text=True).strip()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--revision",default="HEAD")
    a=p.parse_args()
    if not re.fullmatch(r"(?:HEAD|[a-f0-9]{7,40})",a.revision):
        p.error("HEAD or explicit commit SHA required")
    root=Path(git("rev-parse","--show-toplevel")).resolve(strict=True)
    revision=git("rev-parse","--verify",a.revision+"^{commit}",cwd=root)
    base=(root/".run/pinned-code").resolve()
    if not base.is_relative_to(root):p.error("Snapshot folder escaped project")
    base.mkdir(parents=True,exist_ok=True);target=base/revision[:12]
    environment=Path(os.environ.get("UV_PROJECT_ENVIRONMENT",str(root/".venv"))).resolve(strict=True)
    python=environment/"bin/python"
    if not python.is_file():p.error("Existing Linux uv environment required; no environment copying")
    # Hold shared objects against ordinary primary-repository GC.
    subprocess.run(["git","update-ref","refs/experiment-pins/"+revision,revision],cwd=root,check=True)
    if not target.exists():
        subprocess.run(["git","clone","--quiet","--shared","--no-checkout",str(root),str(target)],check=True)
        subprocess.run(["git","checkout","--quiet","--detach",revision],cwd=target,check=True)
    if git("rev-parse","HEAD",cwd=target)!=revision:
        raise ValueError("Existing snapshot has another revision")
    subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=target,check=True)
    env=os.environ.copy();env["PYTHONPATH"]=str(target/"src");env["UV_PROJECT_ENVIRONMENT"]=str(environment)
    observed=subprocess.check_output([str(python),"-c",
        "import pathlib,watermark_studio;print(pathlib.Path(watermark_studio.__file__).resolve())"],
        cwd=target,env=env,text=True).strip()
    if not Path(observed).is_relative_to(target/"src"):
        raise ValueError("Editable installation bypassed the pinned source")
    report={"revision":revision,"checkout":str(target),"pythonpath":str(target/"src"),
        "uv_environment":str(environment),"verified_import":observed,
        "source_transport":"Local Git objects after the authorized push/pull synchronization",
        "scope":"Fixed experiment checkout; never pull into this checkout or sync its shared uv environment"}
    (target/".run").mkdir(exist_ok=True)
    (target/".run/pin.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=="__main__":
    main()
