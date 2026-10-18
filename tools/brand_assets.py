"""Brand assets: render the PNG / ICO files from the SVG masters, and check them.

    python -m tools.brand_assets               # check: brand.json valid, every asset present (no Qt needed)
    python -m tools.brand_assets --build       # re-render PNG + ICO from the SVGs (build time only, needs PyQt5.QtSvg)
    python -m tools.brand_assets --preview P   # also write a light / dark / 16-64 px preview sheet to P

The SVGs in app_assets/brand/ are the masters. The rendered files are kept next to them, so the running application
needs neither Qt nor an internet connection: it only opens PNG / ICO files with Pillow / Tk (ui/brand.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "app_assets" / "brand"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
KEYS = ("name", "descriptor", "tagline", "application", "version", "assets")


def load() -> dict:
    return json.loads((BRAND / "brand.json").read_text(encoding="utf-8"))


def problems() -> list:
    """Everything wrong with the brand folder, in words. Empty = OK."""
    out = []
    try:
        b = load()
    except (OSError, ValueError) as e:
        return [f"brand.json unreadable: {e}"]
    out += [f"brand.json: missing '{k}'" for k in KEYS if k not in b]
    parts = b.get("name_parts")
    if parts and "".join(parts) != b.get("name"):
        out.append("brand.json: name_parts do not spell name (the header would show the old name)")
    for key, fname in (b.get("assets") or {}).items():
        if not (BRAND / fname).is_file():
            out.append(f"asset missing: {key} = {fname} (run python -m tools.brand_assets --build)")
    return out


def _render(svg: Path, w: int, h: int):
    """SVG -> PIL RGBA image of exactly w x h, transparent background, via Qt's SVG renderer."""
    from PyQt5 import QtCore, QtGui, QtSvg
    from PIL import Image
    r = QtSvg.QSvgRenderer(str(svg))
    if not r.isValid():
        raise ValueError(f"{svg.name}: not a valid SVG for Qt")
    img = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
    img.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setRenderHint(QtGui.QPainter.TextAntialiasing)
    p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
    r.render(p, QtCore.QRectF(0, 0, w, h))
    p.end()
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.WriteOnly)
    img.save(buf, "PNG")
    import io
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA")


def build() -> list:
    from PyQt5 import QtGui
    _app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([sys.argv[0]])   # Qt needs one for fonts
    a = load()["assets"]
    written = []
    for svg_key, png_key in (("logo_svg", "logo_png"), ("logo_dark_svg", "logo_dark_png")):
        _render(BRAND / a[svg_key], 2320, 520).save(BRAND / a[png_key])          # 2x the 1160 x 260 viewBox
        written.append(a[png_key])
    _render(BRAND / a["mark_svg"], 512, 512).save(BRAND / a["mark_png"])
    written.append(a["mark_png"])
    # Each icon size rendered from the vector, not shrunk from the big bitmap: sharper at 16 / 32 px.
    sizes = [_render(BRAND / a["mark_svg"], s, s) for s in ICON_SIZES]
    sizes[-1].save(BRAND / a["icon_ico"], format="ICO", sizes=[(s, s) for s in ICON_SIZES],
                   append_images=sizes[:-1])
    written.append(a["icon_ico"])
    return written


def preview(path: Path) -> None:
    """One sheet: the logo on the light and the dark header colours, and the mark at 16-64 px on both."""
    from PIL import Image
    import importlib
    theme = importlib.import_module("ui.theme")
    a = load()["assets"]
    light = Image.open(BRAND / a["logo_png"]).resize((580, 130), Image.LANCZOS)
    dark = Image.open(BRAND / a["logo_dark_png"]).resize((580, 130), Image.LANCZOS)
    mark = Image.open(BRAND / a["mark_png"])
    sheet = Image.new("RGB", (640, 520), "white")
    for y, bg, logo in ((0, theme.LIGHT["RAIL"], light), (260, theme.DARK["RAIL"], dark)):
        tile = Image.new("RGB", (640, 260), bg)
        tile.paste(logo, (30, 20), logo)
        x = 30
        for s in (16, 24, 32, 48, 64):
            m = mark.resize((s, s), Image.LANCZOS)
            tile.paste(m, (x, 170), m)
            x += s + 24
        sheet.paste(tile, (0, y))
    sheet.save(path)


def main() -> None:
    if "--build" in sys.argv:
        for f in build():
            print("wrote", f)
    if "--preview" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--preview") + 1])
        preview(out)
        print("wrote", out)
    bad = problems()
    for p in bad:
        print("FAIL", p)
    if bad:
        sys.exit(1)
    b = load()
    print(f"ok  brand {b['name']} {b['version']}: {len(b['assets'])} assets present")


if __name__ == "__main__":
    main()
