"""A Windows shortcut that starts the application with the product logo as its icon.

    python -m tools.make_shortcut              # <repo>/NexaInspect.lnk  -> run.bat, icon app_assets/brand/nexainspect.ico
    python -m tools.make_shortcut --desktop    # the same shortcut on this user's Desktop
    python -m tools.make_shortcut --selftest   # builds one in a temp folder and reads it back

The shortcut holds absolute paths of THIS computer, so it is not committed (.gitignore: *.lnk); re-run this after
moving the folder. Name and icon come from ui/brand.py (app_assets/brand/brand.json).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ui import brand  # noqa: E402


def _powershell() -> str:
    """Windows PowerShell by its fixed location (not always on PATH, e.g. under Git Bash), else pwsh."""
    exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    return str(exe) if exe.is_file() else "pwsh"


def _ps(script: str) -> str:
    p = subprocess.run([_powershell(), "-NoProfile", "-NonInteractive", "-Command", script],
                       capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip())
    return p.stdout.strip()


def _q(p) -> str:
    return "'" + str(p).replace("'", "''") + "'"


def make(folder: Path) -> Path:
    if not sys.platform.startswith("win"):
        raise RuntimeError("Windows shortcuts only")
    ico = brand.asset("icon_ico")
    if ico is None:
        raise RuntimeError("icon missing: python -m tools.brand_assets --build")
    lnk = Path(folder) / f"{brand.NAME}.lnk"
    _ps("$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + _q(lnk) + "); "
        f"$s.TargetPath = {_q(ROOT / 'run.bat')}; $s.WorkingDirectory = {_q(ROOT)}; "
        f"$s.IconLocation = {_q(str(ico) + ',0')}; "
        f"$s.Description = {_q(f'{brand.NAME} - {brand.APPLICATION}')}; $s.Save()")
    return lnk


def read(lnk: Path) -> dict:
    out = _ps("$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + _q(lnk) + "); "
              "$s.TargetPath; $s.IconLocation; $s.WorkingDirectory")
    target, icon, wd = (out.splitlines() + ["", "", ""])[:3]
    return {"target": target, "icon": icon, "workdir": wd}


def selftest() -> None:
    with tempfile.TemporaryDirectory() as d:
        lnk = make(Path(d))
        got = read(lnk)
        assert lnk.is_file(), lnk
        assert Path(got["target"]) == ROOT / "run.bat", got
        assert got["icon"].lower().startswith(str(brand.asset("icon_ico")).lower()), got
    print(f"ok  shortcut {brand.NAME}.lnk -> run.bat with the product icon")


def main() -> None:
    if "--selftest" in sys.argv:
        return selftest()
    folder = Path(os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")) if "--desktop" in sys.argv else ROOT
    print("wrote", make(folder))


if __name__ == "__main__":
    main()
