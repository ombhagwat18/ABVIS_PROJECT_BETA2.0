"""One palette for every window: a LIGHT industrial HMI by default, the older dark one on request.

Grey carries the interface; colour carries state. Green, red and amber mean
PASS, REJECT and FAULT/warning everywhere -- in buttons, lamps, card badges,
charts and the OpenCV overlays drawn on video -- so an operator never has to
learn a second meaning for a colour. Blue is reserved for selection, focus and
the one primary action on a page.

The palette is chosen ONCE, at import, from settings.json "ui_theme" ("light" | "dark"); a change
applies at the next start (like the text size: re-colouring a built window means redrawing every
widget). Camera images always sit on a dark background (VIDEO_BG) in both themes -- a bright frame
around a video makes the image look dark.

Every module imports from here (gui, hmi, annotation_studio, charts, infer). No
module defines its own hex values: two palettes always drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path

LIGHT = {
    # surfaces
    "BG": "#eef1f4", "PANEL": "#ffffff", "PANEL_2": "#e6eaef", "RAIL": "#dfe4ea", "LINE": "#c3cad3",
    "VIDEO_BG": "#1c2026", "FIELD": "#ffffff", "HOVER": "#d5dbe3", "HOVER_2": "#c8d0da",
    # text
    "INK": "#111827", "DIM": "#4b5563", "MUTED": "#8b95a1",
    # accent
    "ACC": "#1d4ed8", "ACC_H": "#1e40af", "ACC_T": "#ffffff", "ACC_SOFT": "#dbe6fd",
    # state
    "PASS": "#15803d", "REJECT": "#c81e1e", "FAULT": "#b45309", "INFO": "#0369a1", "OFF": "#9ca3af",
    "PASS_SOFT": "#dcfce7", "REJECT_SOFT": "#fee2e2", "FAULT_SOFT": "#fef3c7", "INFO_SOFT": "#e0f2fe",
}
DARK = {
    "BG": "#15181c", "PANEL": "#1e2329", "PANEL_2": "#272d35", "RAIL": "#101316", "LINE": "#343b44",
    "VIDEO_BG": "#0b0d10", "FIELD": "#0b0d10", "HOVER": "#323a44", "HOVER_2": "#3d4652",
    "INK": "#e6e9ed", "DIM": "#8b95a1", "MUTED": "#5d6773",
    "ACC": "#2f6fed", "ACC_H": "#2559c4", "ACC_T": "#ffffff", "ACC_SOFT": "#1f3354",
    "PASS": "#22c55e", "REJECT": "#ef4444", "FAULT": "#f59e0b", "INFO": "#38bdf8", "OFF": "#4b5563",
    "PASS_SOFT": "#163524", "REJECT_SOFT": "#3d1b1b", "FAULT_SOFT": "#3b2c10", "INFO_SOFT": "#12303f",
}
SERIES_LIGHT = ["#2563eb", "#7c3aed", "#0d9488", "#db2777", "#0891b2", "#9333ea", "#0f766e"]
SERIES_DARK = ["#60a5fa", "#a78bfa", "#2dd4bf", "#f472b6", "#93c5fd", "#c4b5fd", "#5eead4"]


def _chosen() -> str:
    try:
        v = json.loads((Path(__file__).resolve().parent.parent / "settings.json").read_text(encoding="utf-8")).get("ui_theme")
    except (OSError, ValueError):
        v = None
    return "dark" if v == "dark" else "light"


MODE = _chosen()
_P = DARK if MODE == "dark" else LIGHT

# ---------------------------------------------------------------- surfaces
BG = _P["BG"]                # window background
PANEL = _P["PANEL"]          # cards, bars, group boxes
PANEL_2 = _P["PANEL_2"]      # raised / hovered / selected rows inside a panel
RAIL = _P["RAIL"]            # navigation rail and status bar
LINE = _P["LINE"]            # borders, gridlines, separators
VIDEO_BG = _P["VIDEO_BG"]    # behind camera frames and image previews (dark in both themes)
FIELD = _P["FIELD"]          # entry / text box background
HOVER, HOVER_2 = _P["HOVER"], _P["HOVER_2"]

# ---------------------------------------------------------------- text
INK = _P["INK"]              # primary text
DIM = _P["DIM"]              # secondary text, captions, axis labels
MUTED = _P["MUTED"]          # disabled text, "no frame" placeholders

# ---------------------------------------------------------------- accent
ACC = _P["ACC"]              # selection, focus, the primary button on a page
ACC_H = _P["ACC_H"]          # its hover
ACC_T = _P["ACC_T"]          # text on ACC (and on any state-coloured button)
ACC_SOFT = _P["ACC_SOFT"]    # a selected row / active item background

# ---------------------------------------------------------------- state (meaning, never decoration)
PASS = _P["PASS"]
REJECT = _P["REJECT"]
FAULT = _P["FAULT"]          # also: warning, "new / not reviewed"
INFO = _P["INFO"]            # AI suggestion, informational
OFF = _P["OFF"]              # a lamp or bit that is off / unknown

# historical names the GUI code already uses
GOOD, BAD, WARN = PASS, REJECT, FAULT

# Soft fills for state backgrounds (confusion matrix cells, banners).
PASS_SOFT, REJECT_SOFT, FAULT_SOFT, INFO_SOFT = _P["PASS_SOFT"], _P["REJECT_SOFT"], _P["FAULT_SOFT"], _P["INFO_SOFT"]

# Series colours for line charts, distinguishable on PANEL, none of them a state colour.
SERIES = SERIES_DARK if MODE == "dark" else SERIES_LIGHT

# ---------------------------------------------------------------- fonts
UI = "Segoe UI"
MONO_FACE = "Consolas"
H1 = (UI, 19, "bold")
H2 = (UI, 15, "bold")
BODY = (UI, 14)
SMALL = (UI, 13)
TINY = (UI, 12)
CAPS = (UI, 12, "bold")   # rail group headers, lamp labels
MONO = (MONO_FACE, 14)
BIG = (UI, 26, "bold")    # machine state / result banners, readable across the room
HUGE = (UI, 34, "bold")


def bgr(hex_colour: str) -> tuple:
    """'#rrggbb' -> OpenCV (b, g, r)."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def apply_ctk(ctk) -> None:
    """Appearance mode, and the stock widget colours pulled into this palette, so a
    widget created without explicit colours still matches."""
    ctk.set_appearance_mode("dark" if MODE == "dark" else "light")
    ctk.set_default_color_theme("dark-blue")
    t = ctk.ThemeManager.theme
    # Widgets that name no font (buttons, entries, menus, plain labels) fall back to this one. Stock is Roboto 13,
    # which is small on a 1080p shop-floor monitor and is not installed on most Windows machines.
    t["CTkFont"] = {"family": UI, "size": 15, "weight": "normal"}
    _guard_scrollbar(ctk)
    pairs = {
        "CTk": {"fg_color": [BG, BG]},
        "CTkToplevel": {"fg_color": [BG, BG]},
        "CTkFrame": {"fg_color": [PANEL, PANEL], "top_fg_color": [PANEL_2, PANEL_2], "border_color": [LINE, LINE]},
        "CTkButton": {"fg_color": [PANEL_2, PANEL_2], "hover_color": [HOVER, HOVER],
                      "border_color": [LINE, LINE], "text_color": [INK, INK],
                      "text_color_disabled": [MUTED, MUTED]},
        "CTkLabel": {"text_color": [INK, INK]},
        "CTkEntry": {"fg_color": [FIELD, FIELD], "border_color": [LINE, LINE], "text_color": [INK, INK],
                     "placeholder_text_color": [MUTED, MUTED]},
        "CTkOptionMenu": {"fg_color": [PANEL_2, PANEL_2], "button_color": [HOVER, HOVER],
                          "button_hover_color": [HOVER_2, HOVER_2], "text_color": [INK, INK]},
        "CTkComboBox": {"fg_color": [FIELD, FIELD], "border_color": [LINE, LINE],
                        "button_color": [HOVER, HOVER], "text_color": [INK, INK]},
        "CTkCheckBox": {"fg_color": [ACC, ACC], "hover_color": [ACC_H, ACC_H], "border_color": [DIM, DIM],
                        "text_color": [INK, INK]},
        "CTkSwitch": {"progress_color": [ACC, ACC], "text_color": [INK, INK], "fg_color": [LINE, LINE]},
        "CTkSlider": {"progress_color": [ACC, ACC], "button_color": [ACC, ACC], "button_hover_color": [ACC_H, ACC_H]},
        "CTkProgressBar": {"progress_color": [ACC, ACC], "fg_color": [LINE, LINE]},
        "CTkTextbox": {"fg_color": [FIELD, FIELD], "text_color": [INK, INK], "border_color": [LINE, LINE]},
        "CTkScrollbar": {"button_color": [HOVER_2, HOVER_2], "button_hover_color": [MUTED, MUTED]},
        "CTkSegmentedButton": {"fg_color": [RAIL, RAIL], "selected_color": [ACC, ACC],
                               "selected_hover_color": [ACC_H, ACC_H], "unselected_color": [PANEL_2, PANEL_2],
                               "unselected_hover_color": [HOVER, HOVER], "text_color": [INK, INK]},
        "CTkScrollableFrame": {"label_fg_color": [PANEL_2, PANEL_2]},
        "DropdownMenu": {"fg_color": [PANEL, PANEL], "hover_color": [ACC_SOFT, ACC_SOFT],
                         "text_color": [INK, INK]},
    }
    for widget, vals in pairs.items():
        t.setdefault(widget, {}).update(vals)


def _guard_scrollbar(ctk) -> None:
    """CTkScrollbar._draw calls update_idletasks(), which can run the pending geometry pass of the scrollable
    frame it belongs to, which calls scrollbar.set() again, which draws again... Rescaling the widgets
    (Settings -> text size) with a scrollable page on screen nested that hundreds of levels deep and the window
    stopped responding. A draw that is already running on a scrollbar is not started again."""
    from customtkinter.windows.widgets.ctk_scrollbar import CTkScrollbar
    if getattr(CTkScrollbar, "_guarded", False):
        return
    orig = CTkScrollbar._draw

    def _draw(self, *a, **k):
        if getattr(self, "_in_draw", False):
            return None
        self._in_draw = True
        try:
            return orig(self, *a, **k)
        finally:
            self._in_draw = False
    CTkScrollbar._draw = _draw
    CTkScrollbar._guarded = True
