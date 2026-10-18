"""The product brand, from ONE file: app_assets/brand/brand.json (name, descriptor, tagline, version, asset files).

Renaming the product = editing brand.json (and, for a new logo, the SVG masters + python -m tools.brand_assets --build).
No screen hard-codes the name. Only local PNG / ICO files are opened (never the network, never Qt at run time); a
missing or broken file falls back to the built-in text so the application still starts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "app_assets" / "brand"
DEFAULT = {"name": "NexaInspect", "name_parts": ["Nexa", "Inspect"],
           "descriptor": "Industrial Vision Inspection Platform", "tagline": "Inspect. Decide. Trace.",
           "application": "Bottle Defect Detection System", "version": "0.9.0", "version_note": "", "assets": {}}


def _load() -> dict:
    try:
        got = json.loads((FOLDER / "brand.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        got = {}
    return {**DEFAULT, **{k: v for k, v in got.items() if v not in (None, "")}}


INFO = _load()
NAME = str(INFO["name"])
DESCRIPTOR = str(INFO["descriptor"])
TAGLINE = str(INFO["tagline"])
APPLICATION = str(INFO["application"])
VERSION = str(INFO["version"])


def name_parts() -> list:
    """The two-tone wordmark ("Nexa" + "Inspect"); the plain name when name_parts no longer spell it."""
    parts = INFO.get("name_parts") or []
    return list(parts) if parts and "".join(parts) == NAME else [NAME]


def asset(key: str):
    """Path of an asset named in brand.json, or None when it is not configured / not on disk."""
    fname = (INFO.get("assets") or {}).get(key)
    p = FOLDER / fname if fname else None
    return p if p is not None and p.is_file() else None


def window_title() -> str:
    return f"{NAME}  -  {APPLICATION}"


def ctk_image(key: str, height: int):
    """A CTkImage of the asset, `height` px high, aspect kept (never stretched). None if the file is unusable."""
    p = asset(key)
    if p is None:
        return None
    try:
        import customtkinter as ctk
        from PIL import Image
        img = Image.open(p)
        img.load()
        return ctk.CTkImage(light_image=img, dark_image=img, size=(max(1, round(height * img.width / img.height)), height))
    except Exception:
        return None


def logo_key(mode: str) -> str:
    """The full logo drawn for this theme: the light one has a navy word that vanishes on a dark surface."""
    return "logo_dark_png" if mode == "dark" else "logo_png"


def install_icon(ctk) -> None:
    """Every CTk / CTkToplevel window (main window AND dialogs) gets the product icon instead of CustomTkinter's.
    CustomTkinter sets its own icon ~200 ms after a window opens unless iconbitmap() was called; we replace that
    one hook, so a window that sets its own icon keeps it."""
    ico = asset("icon_ico")
    if ico is None or not sys.platform.startswith("win"):
        return
    # Without its own AppUserModelID the process is grouped under python.exe and the TASKBAR shows the Python logo
    # even though the window has ours.
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"{NAME}.InspectionApp")
    except Exception:
        pass
    for cls in (ctk.CTk, ctk.CTkToplevel):
        if getattr(cls, "_brand_icon", False):
            continue

        def _set(self, _ico=str(ico)):
            try:
                if not self._iconbitmap_method_called:
                    self.iconbitmap(_ico)
            except Exception:
                pass
        cls._windows_set_titlebar_icon = _set
        cls._brand_icon = True


def selftest() -> None:
    global INFO
    assert NAME and DESCRIPTOR and VERSION, INFO
    assert "".join(name_parts()) == NAME
    for key in ("logo_png", "logo_dark_png", "mark_png", "icon_ico"):
        assert asset(key) is not None, f"{key} missing: python -m tools.brand_assets --build"
    assert asset("no_such_key") is None
    assert logo_key("dark") == "logo_dark_png" and logo_key("light") == "logo_png"
    # renamed in brand.json but the two-tone parts forgotten: the header shows the new name, never the old one
    keep = INFO
    try:
        INFO = {**keep, "name_parts": ["Old", "Name"]}
        assert name_parts() == [NAME], name_parts()
    finally:
        INFO = keep
    print(f"ok  brand {NAME} {VERSION}: wordmark {name_parts()}, logo + mark + icon present")


if __name__ == "__main__":
    selftest()
