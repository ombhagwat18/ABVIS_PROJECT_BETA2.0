"""One palette for every window: a dark industrial HMI.

Grey carries the interface; colour carries state. Green, red and amber mean
PASS, REJECT and FAULT/warning everywhere -- in buttons, lamps, card badges,
charts and the OpenCV overlays drawn on video -- so an operator never has to
learn a second meaning for a colour. Blue is reserved for selection, focus and
the one primary action on a page.

Every module imports from here (gui, annotation_studio, charts, infer). No
module defines its own hex values: two palettes always drift apart.
"""
from __future__ import annotations

# ---------------------------------------------------------------- surfaces
BG = "#15181c"            # window background
PANEL = "#1e2329"         # cards, bars, group boxes
PANEL_2 = "#272d35"       # raised / hovered / selected rows inside a panel
RAIL = "#101316"          # navigation rail and status bar
LINE = "#343b44"          # borders, gridlines, separators
VIDEO_BG = "#0b0d10"      # behind camera frames and image previews

# ---------------------------------------------------------------- text
INK = "#e6e9ed"           # primary text
DIM = "#8b95a1"           # secondary text, captions, axis labels
MUTED = "#5d6773"         # disabled text, "no frame" placeholders

# ---------------------------------------------------------------- accent
ACC = "#2f6fed"           # selection, focus, the primary button on a page
ACC_H = "#2559c4"         # its hover
ACC_T = "#ffffff"         # text on ACC
ACC_SOFT = "#1f3354"      # a selected row / active item background

# ---------------------------------------------------------------- state (meaning, never decoration)
PASS = "#22c55e"
REJECT = "#ef4444"
FAULT = "#f59e0b"         # also: warning, "new / not reviewed"
INFO = "#38bdf8"          # AI suggestion, informational
OFF = "#4b5563"           # a lamp or bit that is off / unknown

# historical names the GUI code already uses
GOOD, BAD, WARN = PASS, REJECT, FAULT

# Soft fills for state backgrounds (confusion matrix cells, banners).
PASS_SOFT, REJECT_SOFT, FAULT_SOFT, INFO_SOFT = "#163524", "#3d1b1b", "#3b2c10", "#12303f"

# Series colours for line charts, distinguishable on PANEL, none of them a state colour.
SERIES = ["#60a5fa", "#a78bfa", "#2dd4bf", "#f472b6", "#93c5fd", "#c4b5fd", "#5eead4"]

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


def bgr(hex_colour: str) -> tuple:
    """'#rrggbb' -> OpenCV (b, g, r)."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def apply_ctk(ctk) -> None:
    """Dark mode, and the stock widget colours pulled into this palette, so a
    widget created without explicit colours still matches."""
    ctk.set_appearance_mode("dark")
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
        "CTkButton": {"fg_color": [PANEL_2, PANEL_2], "hover_color": ["#323a44", "#323a44"],
                      "border_color": [LINE, LINE], "text_color": [INK, INK],
                      "text_color_disabled": [MUTED, MUTED]},
        "CTkLabel": {"text_color": [INK, INK]},
        "CTkEntry": {"fg_color": [VIDEO_BG, VIDEO_BG], "border_color": [LINE, LINE], "text_color": [INK, INK],
                     "placeholder_text_color": [MUTED, MUTED]},
        "CTkOptionMenu": {"fg_color": [PANEL_2, PANEL_2], "button_color": ["#323a44", "#323a44"],
                          "button_hover_color": ["#3d4652", "#3d4652"], "text_color": [INK, INK]},
        "CTkComboBox": {"fg_color": [VIDEO_BG, VIDEO_BG], "border_color": [LINE, LINE],
                        "button_color": ["#323a44", "#323a44"], "text_color": [INK, INK]},
        "CTkCheckBox": {"fg_color": [ACC, ACC], "hover_color": [ACC_H, ACC_H], "border_color": [DIM, DIM],
                        "text_color": [INK, INK]},
        "CTkSwitch": {"progress_color": [ACC, ACC], "text_color": [INK, INK]},
        "CTkSlider": {"progress_color": [ACC, ACC], "button_color": [ACC, ACC], "button_hover_color": [ACC_H, ACC_H]},
        "CTkProgressBar": {"progress_color": [ACC, ACC], "fg_color": [LINE, LINE]},
        "CTkTextbox": {"fg_color": [VIDEO_BG, VIDEO_BG], "text_color": [INK, INK], "border_color": [LINE, LINE]},
        "CTkScrollbar": {"button_color": ["#3a424c", "#3a424c"], "button_hover_color": ["#4a535e", "#4a535e"]},
        "CTkSegmentedButton": {"fg_color": [RAIL, RAIL], "selected_color": [ACC, ACC],
                               "selected_hover_color": [ACC_H, ACC_H], "unselected_color": [PANEL_2, PANEL_2],
                               "unselected_hover_color": ["#323a44", "#323a44"], "text_color": [INK, INK]},
        "CTkScrollableFrame": {"label_fg_color": [PANEL_2, PANEL_2]},
        "DropdownMenu": {"fg_color": [PANEL_2, PANEL_2], "hover_color": [ACC_SOFT, ACC_SOFT],
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
