"""Small shared widgets in the NexaInspect look. Colours and fonts come only from ui/theme.py.

Used by the Annotation Studio first; any page may use them. Keep this module small: a widget belongs here only
when a second place needs the same thing.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from ui import theme
from ui.theme import ACC, ACC_H, ACC_SOFT, ACC_T, BAD, DIM, FAULT, GOOD, INK, LINE, MUTED, OFF, PANEL, PANEL_2

_KIND = {   # fg, hover, text, border
    "primary": (ACC, ACC_H, ACC_T, ACC),
    "secondary": ("transparent", PANEL_2, INK, LINE),
    "success": ("transparent", PANEL_2, GOOD, GOOD),
    "danger": ("transparent", PANEL_2, BAD, BAD),
    "ghost": ("transparent", PANEL_2, ACC, PANEL),
}


def button(parent, text, command, kind: str = "secondary", width: int = 0, tip: str = "", **kw):
    fg, hov, txt, border = _KIND[kind]
    b = ctk.CTkButton(parent, text=text, command=command, fg_color=fg, hover_color=hov, text_color=txt,
                      border_color=border, border_width=0 if kind in ("primary", "ghost") else 1,
                      text_color_disabled=MUTED, height=30, corner_radius=4, **({"width": width} if width else {}), **kw)
    if tip:
        Tooltip(b, tip)
    return b


def section(parent, title: str, padx=12, pady=(10, 4)):
    lab = ctk.CTkLabel(parent, text=title.upper(), text_color=DIM, font=theme.CAPS, anchor="w")
    lab.pack(fill="x", padx=padx, pady=pady)
    return lab


class MetricCard(ctk.CTkFrame):
    """A compact number with a caption (counts are always read from stored state by the caller)."""

    def __init__(self, parent, caption: str, width: int = 96):
        super().__init__(parent, fg_color=PANEL, corner_radius=6, border_width=1, border_color=LINE, width=width,
                         height=54)
        self.pack_propagate(False)
        self.val = ctk.CTkLabel(self, text="-", font=(theme.UI, 18, "bold"), text_color=INK, height=24)
        self.val.pack(anchor="w", padx=10, pady=(6, 0))
        ctk.CTkLabel(self, text=caption.upper(), font=(theme.UI, 10, "bold"), text_color=DIM, height=14
                     ).pack(anchor="w", padx=10)

    def set(self, value, colour: str = INK):
        self.val.configure(text=str(value), text_color=colour)


class StatusPill(ctk.CTkLabel):
    """One-word state of a page or a job: IDLE / GENERATING / PAUSED / REVIEWING / EXPORTING / ERROR ..."""

    LOOK = {"IDLE": OFF, "REVIEWING": GOOD, "GENERATING": ACC, "EXPORTING": ACC, "PAUSED": FAULT,
            "CANCELLING": FAULT, "ERROR": BAD}

    def __init__(self, parent):
        super().__init__(parent, text="IDLE", font=(theme.UI, 12, "bold"), text_color=ACC_T, fg_color=OFF,
                         corner_radius=10, width=110, height=24)
        self.state = "IDLE"

    def set(self, state: str):
        self.state = state
        self.configure(text=state, fg_color=self.LOOK.get(state, OFF))


class Tooltip:
    """Hover help (used for keyboard shortcuts). Plain Tk: shown after a short delay, gone on leave / click."""

    def __init__(self, widget, text: str, delay_ms: int = 600):
        self.w, self.text, self.delay, self.tip, self._job = widget, text, delay_ms, None, None
        widget.bind("<Enter>", self._arm, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _arm(self, _e=None):
        self._job = self.w.after(self.delay, self._show)

    def _show(self):
        if self.tip is not None or not self.w.winfo_exists():
            return
        x, y = self.w.winfo_rootx() + 10, self.w.winfo_rooty() + self.w.winfo_height() + 4
        self.tip = tk.Toplevel(self.w)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, bg=theme.NAVY if theme.MODE == "light" else PANEL_2, fg="#ffffff",
                 font=(theme.UI, 10), padx=8, pady=4, justify="left").pack()

    def _hide(self, _e=None):
        if self._job is not None:
            try:
                self.w.after_cancel(self._job)
            except Exception:                                    # noqa: BLE001
                pass
            self._job = None
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


def confirm(title: str, text: str) -> bool:
    return bool(messagebox.askyesno(title, text))


def empty_state(parent, title: str, body: str):
    f = ctk.CTkFrame(parent, fg_color="transparent")
    ctk.CTkLabel(f, text=title, font=theme.H2, text_color=INK).pack(pady=(40, 6))
    ctk.CTkLabel(f, text=body, text_color=DIM, wraplength=560, justify="center").pack()
    return f


SOFT = {"pending": theme.FAULT_SOFT, "accepted": theme.PASS_SOFT, "corrected": theme.INFO_SOFT,
        "rejected": theme.REJECT_SOFT, "failed": theme.REJECT_SOFT, "selected": ACC_SOFT}
