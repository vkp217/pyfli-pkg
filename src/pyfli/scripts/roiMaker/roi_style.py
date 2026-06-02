"""
roi_style.py
============
Qt stylesheet (QSS) for the ROI Maker window.

Usage
-----
    from .roi_style import STYLE          # full stylesheet (default)
    from .roi_style import BUTTONS, SLIDERS  # individual sections

To customise a section without touching the rest, override just that constant
before constructing ROIApp, then rebuild STYLE:

    import pyfli.scripts.roiMaker.roi_style as S
    S.BUTTONS = S.BUTTONS.replace("#1e66f5", "#e94560")  # swap accent colour
    S.STYLE   = S.BASE + S.SIDEBAR + S.BUTTONS + S.SLIDERS + S.DIALOG + S.STATUSBAR

Colour tokens (Catppuccin Mocha palette)
-----------------------------------------
    #1e1e2e  — base (window background)
    #181825  — mantle (sidebar background)
    #24273a  — crust (button default background)
    #313244  — surface 0 (borders, dividers)
    #363a4f  — surface 1 (hover)
    #45475a  — surface 2 (active border)
    #585b70  — overlay 0 (muted text)
    #a6adc8  — subtext 0 (normal text)
    #cdd6f4  — text (primary)
    #cba6f7  — mauve (accent title)
    #1e66f5  — blue (active / checked)
    #a6e3a1  — green (save text)
    #f38ba8  — red (cancel text)
    #7c3aed  — purple (toggle-on action)
"""

# ── Base window / widget ──────────────────────────────────────────────────────
BASE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
}
"""

# ── Sidebar panel ─────────────────────────────────────────────────────────────
SIDEBAR = """
QWidget#sidebar {
    background-color: #181825;
    border-right: 1px solid #313244;
}
QLabel#section_lbl {
    color: #585b70;
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 1.5px;
}
QLabel#title_lbl {
    color: #cba6f7;
    font-size: 15px;
    font-weight: bold;
    padding: 4px 0px;
}
QFrame#divider {
    background-color: #313244;
    max-height: 1px;
    min-height: 1px;
}
"""

# ── Buttons ───────────────────────────────────────────────────────────────────
BUTTONS = """
/* Tool buttons and mask-type buttons (checkable, exclusive) */
QPushButton#tool_btn {
    background-color: #24273a;
    color: #a6adc8;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 8px 10px;
    text-align: left;
}
QPushButton#tool_btn:hover {
    background-color: #363a4f;
    color: #cdd6f4;
    border-color: #45475a;
}
QPushButton#tool_btn:checked {
    background-color: #1e66f5;
    color: #ffffff;
    border-color: #1d62e8;
    font-weight: bold;
}

/* Action buttons (toggle-able) */
QPushButton#action_btn {
    background-color: #24273a;
    color: #a6adc8;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 8px 10px;
    text-align: left;
}
QPushButton#action_btn:hover {
    background-color: #363a4f;
    color: #cdd6f4;
}
QPushButton#action_btn:checked {
    background-color: #7c3aed;
    color: #ffffff;
    border-color: #6d28d9;
    font-weight: bold;
}

/* Save button */
QPushButton#save_btn {
    background-color: #1e4a2a;
    color: #a6e3a1;
    border: 1px solid #2a6b3a;
    border-radius: 8px;
    padding: 9px 10px;
    text-align: left;
    font-weight: bold;
}
QPushButton#save_btn:hover {
    background-color: #2a6b3a;
    color: #ffffff;
}

/* Cancel button */
QPushButton#cancel_btn {
    background-color: #2a1a1a;
    color: #f38ba8;
    border: 1px solid #4a2525;
    border-radius: 8px;
    padding: 9px 10px;
    text-align: left;
}
QPushButton#cancel_btn:hover {
    background-color: #4a2525;
    color: #ffffff;
}
"""

# ── Sliders ───────────────────────────────────────────────────────────────────
SLIDERS = """
QSlider::groove:horizontal {
    height: 4px;
    background: #313244;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #1e66f5;
    border: none;
    width: 13px;
    height: 13px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #1e66f5;
    border-radius: 2px;
}
"""

# ── Dialog, table, spinbox ────────────────────────────────────────────────────
DIALOG = """
QDialog {
    background-color: #1e1e2e;
}
QTableWidget {
    background-color: #24273a;
    gridline-color: #313244;
    border: 1px solid #313244;
    border-radius: 6px;
    color: #cdd6f4;
}
QTableWidget::item:selected {
    background-color: #363a4f;
}
QHeaderView::section {
    background-color: #181825;
    color: #585b70;
    border: none;
    padding: 4px;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
}
QSpinBox {
    background-color: #24273a;
    color: #cdd6f4;
    border: 1px solid #313244;
    border-radius: 6px;
    padding: 3px 6px;
}
QSpinBox::up-button, QSpinBox::down-button {
    background: #363a4f;
    border: none;
    width: 16px;
}
QDialogButtonBox QPushButton {
    background-color: #24273a;
    color: #a6adc8;
    border: 1px solid #313244;
    border-radius: 6px;
    padding: 6px 16px;
    min-width: 70px;
}
QDialogButtonBox QPushButton:hover {
    background-color: #363a4f;
    color: #cdd6f4;
}
QDialogButtonBox QPushButton:default {
    background-color: #1e66f5;
    color: #ffffff;
    border-color: #1d62e8;
}
"""

# ── Status bar ────────────────────────────────────────────────────────────────
STATUSBAR = """
QStatusBar {
    background-color: #1e66f5;
    color: #ffffff;
    font-size: 11px;
    padding: 0 8px;
}
QStatusBar::item {
    border: none;
}
"""

# ── Combined (used by ROIApp by default) ──────────────────────────────────────
STYLE = BASE + SIDEBAR + BUTTONS + SLIDERS + DIALOG + STATUSBAR
