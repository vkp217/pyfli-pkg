# scripts/roiMaker/roi_maker.py
"""
ROI Maker — PySide6-based interactive region-of-interest editor.

Public API (unchanged):
    maker = ROIMaker(intensity_2d, save_path="mask.npy")
    multi_mask = maker.draw()
    maker.save_masks()
    maker.get_multi_cluster_mask()
    maker.get_binary_mask()
"""

import numpy as np
import cv2          # importing cv2 overwrites QT_QPA_PLATFORM_PLUGIN_PATH
import os
import sys

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QSizePolicy, QStatusBar, QSpacerItem,
    QDialog, QDialogButtonBox, QSlider, QButtonGroup, QColorDialog,
    QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView,
)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QSize
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPolygonF,
    QPixmap, QImage, QFont, QCursor,
)

try:
    from .roi_style import STYLE as _STYLE      # imported as package
except ImportError:
    from roi_style import STYLE as _STYLE       # run directly as script


# ─────────────────────────────────────────────────────────────────────────────
# ROI data model
# ─────────────────────────────────────────────────────────────────────────────

class ROIObject:
    def __init__(self, pts, roi_id=0):
        self.pts      = np.array(pts, dtype=np.int32)
        self.roi_id   = int(roi_id)
        self.assigned = False       # True once the user explicitly assigns an ID
        self.center   = np.mean(self.pts, axis=0)

    def move(self, dx, dy):
        self.pts   += [int(dx), int(dy)]
        self.center = np.mean(self.pts, axis=0)

    def rotate(self, angle_deg):
        rad  = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        M    = np.array([[c, -s], [s, c]])
        self.pts    = ((self.pts - self.center) @ M.T + self.center).astype(np.int32)
        self.center = np.mean(self.pts, axis=0)

    def scale(self, factor):
        self.pts    = ((self.pts - self.center) * factor + self.center).astype(np.int32)
        self.center = np.mean(self.pts, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-ROI colour palette
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    QColor( 30, 102, 245), QColor( 64, 160,  43), QColor(210,  15,  57),
    QColor(136,  57, 239), QColor(223, 142,  29), QColor(254, 100,  11),
    QColor( 23, 146, 153), QColor(156, 160, 176),
]

def _roi_color(roi_id: int) -> QColor:
    return _PALETTE[(roi_id - 1) % len(_PALETTE)]


# ─────────────────────────────────────────────────────────────────────────────
# ID Assignment Dialog (shown before saving in multi / both mode)
# ─────────────────────────────────────────────────────────────────────────────

class IDAssignDialog(QDialog):
    """Let the user rename/reorder ROI IDs before the final save."""

    def __init__(self, rois: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Region IDs")
        self.setMinimumWidth(360)
        self.setStyleSheet(_STYLE)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        lbl = QLabel("Set the ID for each drawn region.\nIDs must be positive integers (duplicates are allowed).")
        lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        self._table = QTableWidget(len(rois), 3)
        self._table.setHorizontalHeaderLabels(["Color", "Auto ID", "New ID"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.setColumnWidth(0, 28)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

        # Next sequential number for unassigned ROIs
        _next = max((r.roi_id for r in rois if r.assigned), default=0) + 1

        self._spinboxes = []
        for row, roi in enumerate(rois):
            # colour swatch — grey for unassigned, palette colour for assigned
            c = _roi_color(roi.roi_id) if roi.assigned else QColor(120, 120, 130)
            swatch = QWidget()
            swatch.setStyleSheet(
                f"background-color: rgb({c.red()},{c.green()},{c.blue()});"
                "border-radius: 3px; margin: 4px;"
            )
            self._table.setCellWidget(row, 0, swatch)

            # status column: shows current ID or "unassigned"
            status_text = str(roi.roi_id) if roi.assigned else "unassigned"
            id_item = QTableWidgetItem(status_text)
            id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            id_item.setForeground(QBrush(QColor("#585b70") if roi.assigned else QColor("#f38ba8")))
            self._table.setItem(row, 1, id_item)

            # editable new ID — pre-fill with assigned ID or next sequential
            default_id = roi.roi_id if roi.assigned else _next
            if not roi.assigned:
                _next += 1

            spin = QSpinBox()
            spin.setRange(1, 9999)
            spin.setValue(default_id)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setCellWidget(row, 2, spin)
            self._spinboxes.append(spin)

        layout.addWidget(self._table)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_assignments(self) -> dict:
        """Return {row_index: new_id} so callers can update roi.roi_id."""
        return {i: spin.value() for i, spin in enumerate(self._spinboxes)}


# ─────────────────────────────────────────────────────────────────────────────
# Drawing canvas
# ─────────────────────────────────────────────────────────────────────────────

_HANDLE_R = 6
_MIN_BOX  = 6


class ImageCanvas(QWidget):
    roi_changed = Signal()

    def __init__(self, rm, parent=None):
        super().__init__(parent)
        self.rm = rm
        self.selected_idx = -1
        self.mode = 'rect'

        # drawing state
        self._drawing  = False
        self._start_i  = None
        self._free_i   = []

        # move state
        self._moving       = False
        self._mv_start_mi  = None
        self._mv_start_pts = None

        # handle-resize state
        self._resizing      = False
        self._rz_handle     = -1
        self._rz_start_mw   = None
        self._rz_start_bbox = None
        self._rz_start_pts  = None

        self._cur_mw = QPointF(0, 0)

        # transform
        self._scale    = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0

        # base image
        arr = np.asarray(rm.display_base, dtype=np.uint8)
        h, w = arr.shape
        self._pixmap = QPixmap.fromImage(
            QImage(arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        )

        # intensity overlay: stored as a numpy RGBA array so paintEvent can
        # wrap it in a fresh QImage each frame without a copy or GC hazard.
        self._int_overlay = None   # np.ndarray (H, W, 4) uint8 or None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)

    # ── transform ─────────────────────────────────────────────────────────────

    def _recompute_transform(self):
        cw, ch = self.width(), self.height()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        self._scale    = min(cw / iw, ch / ih)
        self._offset_x = (cw - iw * self._scale) / 2
        self._offset_y = (ch - ih * self._scale) / 2

    def _w2i(self, wx, wy):
        return ((wx - self._offset_x) / self._scale,
                (wy - self._offset_y) / self._scale)

    def _i2w(self, ix, iy):
        return (ix * self._scale + self._offset_x,
                iy * self._scale + self._offset_y)

    def _qpt_w2i(self, q):
        x, y = self._w2i(q.x(), q.y())
        return QPointF(x, y)

    def _pts_to_poly_w(self, pts):
        return QPolygonF([
            QPointF(*self._i2w(float(p[0]), float(p[1]))) for p in pts
        ])

    # ── intensity overlay ──────────────────────────────────────────────────────

    def update_intensity_overlay(self):
        """Recompute the RGBA overlay array for out-of-range pixels.

        We store the raw numpy array rather than a QPixmap so that paintEvent
        can wrap it in a QImage each frame. This avoids the deferred-copy bug
        that occurs when QPixmap.fromImage() is called on an inline QImage.
        """
        if not self.rm.intensity_active:
            self._int_overlay = None
            return
        lo, hi  = self.rm.intensity_low, self.rm.intensity_high
        arr     = self.rm._raw_img            # (H, W) float64 — original values
        below   = arr < lo
        above   = arr > hi
        rgba = np.zeros((self.rm.H, self.rm.W, 4), dtype=np.uint8)
        rb, gb, bb = self.rm.mask_below_color
        rgba[below, 0] = rb; rgba[below, 1] = gb; rgba[below, 2] = bb; rgba[below, 3] = 190
        ra, ga, ba = self.rm.mask_above_color
        rgba[above, 0] = ra; rgba[above, 1] = ga; rgba[above, 2] = ba; rgba[above, 3] = 190
        self._int_overlay = rgba              # keep array alive for QImage wrapping

    # ── bounding-box handles ───────────────────────────────────────────────────

    @staticmethod
    def _bbox(pts):
        return (float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))

    def _handle_pos_i(self, pts):
        x0, y0, x1, y1 = self._bbox(pts)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        return [(x0,y0),(mx,y0),(x1,y0),(x1,my),(x1,y1),(mx,y1),(x0,y1),(x0,my)]

    def _hit_handle(self, wpt, roi):
        thresh2 = (_HANDLE_R + 3) ** 2
        for i, (ix, iy) in enumerate(self._handle_pos_i(roi.pts)):
            wx, wy = self._i2w(ix, iy)
            if (wpt.x()-wx)**2 + (wpt.y()-wy)**2 <= thresh2:
                return i
        return -1

    def _hit_roi(self, wpt):
        ix, iy = self._w2i(wpt.x(), wpt.y())
        for i, roi in enumerate(self.rm.rois):
            if cv2.pointPolygonTest(roi.pts.astype(np.float32),
                                    (float(ix), float(iy)), False) >= 0:
                return i
        return -1

    # ── painting ───────────────────────────────────────────────────────────────

    def paintEvent(self, _):
        self._recompute_transform()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        p.fillRect(self.rect(), QColor(18, 18, 30))

        iw, ih = self._pixmap.width(), self._pixmap.height()
        target = QRectF(self._offset_x, self._offset_y,
                        iw * self._scale, ih * self._scale)

        if self.rm.show_bg:
            p.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        # intensity mask overlay — wrap the live numpy array in a QImage each
        # frame so there is no stale-pointer or deferred-copy issue.
        if self.rm.intensity_active and self._int_overlay is not None:
            h, w = self._int_overlay.shape[:2]
            _img = QImage(self._int_overlay.data, w, h, 4 * w,
                          QImage.Format.Format_RGBA8888)
            self._int_img_ref = _img   # prevent GC while QPainter holds it
            p.drawImage(target, _img)

        for i, roi in enumerate(self.rm.rois):
            self._paint_roi(p, roi, selected=(i == self.selected_idx))

        if self._drawing and self._start_i is not None:
            self._paint_preview(p)

        p.end()

    def _paint_roi(self, p, roi, selected):
        if not roi.assigned:
            color      = QColor(0, 220, 100) if selected else QColor(140, 140, 155)
            line_style = Qt.PenStyle.DashLine
            label_text = "?"
        else:
            color      = QColor(0, 220, 100) if selected else _roi_color(roi.roi_id)
            line_style = Qt.PenStyle.SolidLine
            label_text = f"ID:{roi.roi_id}"

        poly = self._pts_to_poly_w(roi.pts)
        fill = QColor(color.red(), color.green(), color.blue(), 40 if not roi.assigned else 55)
        p.setBrush(QBrush(fill))
        pen = QPen(color, 2.5 if selected else 1.5, line_style)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawPolygon(poly)
        if roi.pts.shape[0]:
            wx, wy = self._i2w(float(roi.pts[0, 0]), float(roi.pts[0, 1]))
            p.setPen(QPen(Qt.GlobalColor.white))
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            p.drawText(int(wx) + 4, int(wy) + 13, label_text)
        if selected:
            self._paint_handles(p, roi)

    def _paint_handles(self, p, roi):
        for ix, iy in self._handle_pos_i(roi.pts):
            wx, wy = self._i2w(ix, iy)
            r = _HANDLE_R
            p.setBrush(QBrush(QColor(255, 255, 255, 230)))
            p.setPen(QPen(QColor(30, 100, 220), 1.5))
            p.drawEllipse(QRectF(wx-r, wy-r, r*2, r*2))

    def _paint_preview(self, p):
        pts = self._preview_pts()
        if len(pts) < 3:
            return
        poly = QPolygonF([QPointF(*self._i2w(float(pt[0]), float(pt[1]))) for pt in pts])
        p.setBrush(QBrush(QColor(255, 255, 255, 25)))
        p.setPen(QPen(QColor(255, 255, 255, 160), 1.2, Qt.PenStyle.DashLine))
        p.drawPolygon(poly)

    def _preview_pts(self):
        if self._start_i is None:
            return []
        ix, iy = self._start_i
        mx, my = self._w2i(self._cur_mw.x(), self._cur_mw.y())
        if self.mode == 'rect':
            return [[ix, iy], [mx, iy], [mx, my], [ix, my]]
        if self.mode == 'circle':
            r = max(int(np.hypot(mx-ix, my-iy)), 3)
            return cv2.ellipse2Poly((int(ix), int(iy)), (r, r), 0, 0, 360, 8).tolist()
        return list(self._free_i)

    # ── mouse ──────────────────────────────────────────────────────────────────

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        wpt = e.position()

        # ── Assign mode: click a ROI to give it the next sequential ID ──────
        if self.mode == 'assign':
            hit = self._hit_roi(wpt)
            if hit != -1:
                roi = self.rm.rois[hit]
                if not roi.assigned:
                    roi.roi_id   = self.rm.assign_counter
                    roi.assigned = True
                    self.rm.assign_counter += 1
                self.selected_idx = hit
            else:
                self.selected_idx = -1
            self.update()
            self.roi_changed.emit()
            return
        # ────────────────────────────────────────────────────────────────────

        if self.selected_idx != -1:
            hi = self._hit_handle(wpt, self.rm.rois[self.selected_idx])
            if hi != -1:
                roi = self.rm.rois[self.selected_idx]
                self._resizing      = True
                self._rz_handle     = hi
                self._rz_start_mw   = wpt
                self._rz_start_bbox = self._bbox(roi.pts)
                self._rz_start_pts  = roi.pts.copy().astype(float)
                self.update()
                return

        hit = self._hit_roi(wpt)
        if hit != -1:
            self.selected_idx  = hit
            self._moving       = True
            self._mv_start_mi  = self._qpt_w2i(wpt)
            self._mv_start_pts = self.rm.rois[hit].pts.copy().astype(float)
            self.update(); self.roi_changed.emit(); return

        if self.mode != 'select':
            self.selected_idx = -1
            self._drawing  = True
            ix, iy         = self._w2i(wpt.x(), wpt.y())
            self._start_i  = (ix, iy)
            self._free_i   = [(ix, iy)]
        else:
            self.selected_idx = -1
        self.update(); self.roi_changed.emit()

    def mouseMoveEvent(self, e):
        wpt = e.position()
        self._cur_mw = wpt

        if self._resizing and self.selected_idx != -1:
            self._apply_resize(wpt)
        elif self._moving and self.selected_idx != -1:
            cur_i = self._qpt_w2i(wpt)
            dx = cur_i.x() - self._mv_start_mi.x()
            dy = cur_i.y() - self._mv_start_mi.y()
            roi = self.rm.rois[self.selected_idx]
            new_pts = self._mv_start_pts + np.array([dx, dy])
            roi.pts    = np.clip(new_pts, 0, None).astype(np.int32)
            roi.center = np.mean(roi.pts, axis=0)
        elif self._drawing and self.mode == 'freehand' and self._start_i:
            ix, iy = self._w2i(wpt.x(), wpt.y())
            self._free_i.append((ix, iy))

        # adaptive cursor
        if self.selected_idx != -1 and not self._drawing:
            hi = self._hit_handle(wpt, self.rm.rois[self.selected_idx])
            cursors = {
                0: Qt.CursorShape.SizeFDiagCursor, 4: Qt.CursorShape.SizeFDiagCursor,
                2: Qt.CursorShape.SizeBDiagCursor, 6: Qt.CursorShape.SizeBDiagCursor,
                1: Qt.CursorShape.SizeVerCursor,   5: Qt.CursorShape.SizeVerCursor,
                3: Qt.CursorShape.SizeHorCursor,   7: Qt.CursorShape.SizeHorCursor,
            }
            if hi in cursors:
                self.setCursor(cursors[hi])
            elif self._hit_roi(wpt) != -1:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(
                    Qt.CursorShape.CrossCursor if self.mode != 'select'
                    else Qt.CursorShape.ArrowCursor
                )
        else:
            self.setCursor(
                Qt.CursorShape.CrossCursor if self.mode not in ('select', None)
                else Qt.CursorShape.ArrowCursor
            )

        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if self._resizing:
            self._resizing = False; self.roi_changed.emit()
        if self._moving:
            self._moving = False; self.roi_changed.emit()
        if self._drawing:
            self._drawing = False
            wpt = e.position()
            ix, iy = self._w2i(wpt.x(), wpt.y())
            pts = self._finalise_pts(ix, iy)
            if len(pts) >= 3:
                roi = ROIObject(pts)      # roi_id=0, assigned=False until user assigns
                self.rm.rois.append(roi)
                self.selected_idx = len(self.rm.rois) - 1
            self._start_i = None; self._free_i = []
            self.roi_changed.emit()
        self.update()

    def _finalise_pts(self, mx, my):
        if self._start_i is None:
            return []
        ix, iy = self._start_i
        if self.mode == 'rect':
            return np.array([[ix,iy],[mx,iy],[mx,my],[ix,my]])
        if self.mode == 'circle':
            r = max(int(np.hypot(mx-ix, my-iy)), 3)
            return cv2.ellipse2Poly((int(ix),int(iy)), (r,r), 0, 0, 360, 8)
        pts = np.array(self._free_i)
        return pts if len(pts) >= 3 else []

    def _apply_resize(self, wpt):
        roi = self.rm.rois[self.selected_idx]
        dix = (wpt.x() - self._rz_start_mw.x()) / self._scale
        diy = (wpt.y() - self._rz_start_mw.y()) / self._scale
        x0, y0, x1, y1 = self._rz_start_bbox
        nx0, ny0, nx1, ny1 = x0, y0, x1, y1
        hi = self._rz_handle
        if hi in (0, 6, 7): nx0 = x0 + dix
        if hi in (2, 3, 4): nx1 = x1 + dix
        if hi in (0, 1, 2): ny0 = y0 + diy
        if hi in (4, 5, 6): ny1 = y1 + diy
        if nx1 - nx0 < _MIN_BOX: nx1 = nx0 + _MIN_BOX
        if ny1 - ny0 < _MIN_BOX: ny1 = ny0 + _MIN_BOX
        bw = max(x1 - x0, 1.0); bh = max(y1 - y0, 1.0)
        pts = self._rz_start_pts.copy()
        pts[:, 0] = nx0 + (pts[:, 0] - x0) * (nx1 - nx0) / bw
        pts[:, 1] = ny0 + (pts[:, 1] - y0) * (ny1 - ny0) / bh
        roi.pts    = np.clip(pts, 0, None).astype(np.int32)
        roi.center = np.mean(roi.pts, axis=0)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._delete_selected()

    def _delete_selected(self):
        if 0 <= self.selected_idx < len(self.rm.rois):
            self.rm.rois.pop(self.selected_idx)
            self.selected_idx = -1
            # Recalculate assign_counter so it stays above all existing IDs
            assigned_ids = [r.roi_id for r in self.rm.rois if r.assigned]
            self.rm.assign_counter = (max(assigned_ids) + 1) if assigned_ids else 1
            self.roi_changed.emit(); self.update()


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class ROIApp(QMainWindow):
    def __init__(self, rm):
        super().__init__()
        self.rm = rm
        self.setWindowTitle("ROI Maker")
        self.setStyleSheet(_STYLE)

        central     = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Canvas first so _build_sidebar signals can reference it safely
        self.canvas = ImageCanvas(rm, self)
        self.canvas.roi_changed.connect(self._refresh_status)
        self.canvas.update_intensity_overlay()

        sidebar = self._build_sidebar()
        root_layout.addWidget(sidebar)
        root_layout.addWidget(self.canvas, stretch=1)

        self.setStatusBar(QStatusBar(self))
        self._refresh_status()

    # ── sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: #181825; }"
                             "QScrollBar:vertical { width: 6px; background: #181825; }"
                             "QScrollBar::handle:vertical { background: #313244; border-radius: 3px; }")
        scroll.setFixedWidth(185)

        sb = QWidget()
        sb.setObjectName("sidebar")
        layout = QVBoxLayout(sb)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(4)
        scroll.setWidget(sb)

        # Title
        title = QLabel("⬡  ROI Maker")
        title.setObjectName("title_lbl")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(self._divider())

        # ── TOOLS ──
        layout.addWidget(self._section_label("TOOLS"))
        self._tool_btns = {}
        for key, icon, label, sc in [
            ('select',   '↖', 'Select / Move', 'S'),
            ('rect',     '▭', 'Rectangle',     'R'),
            ('circle',   '◯', 'Circle',        'C'),
            ('freehand', '✏', 'Freehand',      'F'),
            ('assign',   '⊕', 'Assign IDs',    'A'),
        ]:
            btn = self._tool_button(icon, label, sc)
            btn.toggled.connect(lambda on, k=key: self._on_tool_toggled(k, on))
            self._tool_btns[key] = btn
            layout.addWidget(btn)

        reset_btn = self._action_button('↺', 'Reset IDs', '')
        reset_btn.setToolTip("Clear all ID assignments — ROIs return to unassigned state")
        reset_btn.clicked.connect(self._reset_ids)
        layout.addWidget(reset_btn)

        auto_btn = self._action_button('⟳', 'Auto-assign IDs', '')
        auto_btn.setToolTip("Assign sequential IDs to all unassigned ROIs automatically")
        auto_btn.clicked.connect(self._auto_assign_ids)
        layout.addWidget(auto_btn)

        layout.addSpacing(4); layout.addWidget(self._divider())

        # ── OUTPUT TYPE ──
        layout.addWidget(self._section_label("OUTPUT TYPE"))
        self._mask_type_group = QButtonGroup(self)
        self._mask_type_group.setExclusive(True)
        for val, icon, label in [
            ('binary', '◻', 'Binary'),
            ('multi',  '◼', 'Multi-ID'),
            ('both',   '⊞', 'Both'),
        ]:
            btn = QPushButton(f"{icon}  {label}")
            btn.setObjectName("tool_btn")
            btn.setCheckable(True)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setToolTip(f"Save as: {label}")
            self._mask_type_group.addButton(btn)
            layout.addWidget(btn)
            if val == self.rm.mask_type:
                btn.setChecked(True)
        self._mask_type_group.buttonClicked.connect(self._on_mask_type_btn)
        # store reference to buttons by order
        self._mask_type_btns = {
            v: self._mask_type_group.buttons()[i]
            for i, v in enumerate(['binary', 'multi', 'both'])
        }

        layout.addSpacing(4); layout.addWidget(self._divider())

        # ── EDIT ──
        layout.addWidget(self._section_label("EDIT"))
        self._bg_btn = self._action_button('◑', 'Toggle Image', 'B')
        self._bg_btn.clicked.connect(self._toggle_bg)
        layout.addWidget(self._bg_btn)

        del_btn = self._action_button('✕', 'Delete ROI', 'Del')
        del_btn.clicked.connect(self._delete_selected)
        layout.addWidget(del_btn)

        del_all_btn = self._action_button('⊘', 'Delete All ROIs', '')
        del_all_btn.setToolTip("Remove every ROI and start fresh")
        del_all_btn.clicked.connect(self._delete_all)
        layout.addWidget(del_all_btn)

        layout.addSpacing(4); layout.addWidget(self._divider())

        # ── INTENSITY FILTER ──
        layout.addWidget(self._section_label("INTENSITY FILTER"))

        self._int_btn = QPushButton("⚡  Enable Filter")
        self._int_btn.setObjectName("action_btn")
        self._int_btn.setCheckable(True)
        self._int_btn.setChecked(self.rm.intensity_active)
        self._int_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._int_btn.toggled.connect(self._on_intensity_toggled)
        layout.addWidget(self._int_btn)

        # Low threshold
        self._lo_lbl = QLabel("Low threshold")
        self._lo_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(self._lo_lbl)
        _lo_row = QWidget(); _lo_rl = QHBoxLayout(_lo_row)
        _lo_rl.setContentsMargins(0, 0, 0, 0); _lo_rl.setSpacing(4)
        self._lo_slider = QSlider(Qt.Orientation.Horizontal)
        self._lo_slider.setRange(self.rm.img_min, self.rm.img_max)
        self._lo_slider.setValue(self.rm.intensity_low)
        self._lo_slider.valueChanged.connect(self._on_lo_changed)
        _lo_rl.addWidget(self._lo_slider)
        self._lo_spin = QSpinBox()
        self._lo_spin.setRange(self.rm.img_min, self.rm.img_max)
        self._lo_spin.setValue(self.rm.intensity_low)
        self._lo_spin.setFixedWidth(58)
        self._lo_spin.valueChanged.connect(self._on_lo_spin_changed)
        _lo_rl.addWidget(self._lo_spin)
        layout.addWidget(_lo_row)

        # High threshold
        self._hi_lbl = QLabel("High threshold")
        self._hi_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(self._hi_lbl)
        _hi_row = QWidget(); _hi_rl = QHBoxLayout(_hi_row)
        _hi_rl.setContentsMargins(0, 0, 0, 0); _hi_rl.setSpacing(4)
        self._hi_slider = QSlider(Qt.Orientation.Horizontal)
        self._hi_slider.setRange(self.rm.img_min, self.rm.img_max)
        self._hi_slider.setValue(self.rm.intensity_high)
        self._hi_slider.valueChanged.connect(self._on_hi_changed)
        _hi_rl.addWidget(self._hi_slider)
        self._hi_spin = QSpinBox()
        self._hi_spin.setRange(self.rm.img_min, self.rm.img_max)
        self._hi_spin.setValue(self.rm.intensity_high)
        self._hi_spin.setFixedWidth(58)
        self._hi_spin.valueChanged.connect(self._on_hi_spin_changed)
        _hi_rl.addWidget(self._hi_spin)
        layout.addWidget(_hi_row)

        # Mask-color pickers (separate for below/above)
        self._below_color_btn = QPushButton("▼  Below color")
        self._below_color_btn.setObjectName("action_btn")
        self._below_color_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._below_color_btn.setToolTip("Color for pixels below low threshold")
        self._below_color_btn.clicked.connect(self._pick_below_color)
        self._refresh_below_color_btn()
        layout.addWidget(self._below_color_btn)

        self._above_color_btn = QPushButton("▲  Above color")
        self._above_color_btn.setObjectName("action_btn")
        self._above_color_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._above_color_btn.setToolTip("Color for pixels above high threshold")
        self._above_color_btn.clicked.connect(self._pick_above_color)
        self._refresh_above_color_btn()
        layout.addWidget(self._above_color_btn)

        thresh_btn = self._action_button('⊞', 'Create Threshold ROI', '')
        thresh_btn.setToolTip(
            "Convert the current intensity threshold into ROI(s).\n"
            "They are added as unassigned ROIs — assign IDs the same way as drawn ROIs."
        )
        thresh_btn.clicked.connect(self._create_threshold_rois)
        layout.addWidget(thresh_btn)

        layout.addSpacing(4); layout.addWidget(self._divider())

        # ── FILE ──
        layout.addWidget(self._section_label("FILE"))

        save_btn = QPushButton("✓  Save && Close  ↵")
        save_btn.setObjectName("save_btn")
        save_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        save_btn.clicked.connect(self._save_close)
        layout.addWidget(save_btn)

        cancel_btn = QPushButton("✗  Cancel  Esc")
        cancel_btn.setObjectName("cancel_btn")
        cancel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        cancel_btn.clicked.connect(self._cancel)
        layout.addWidget(cancel_btn)

        layout.addSpacing(4); layout.addWidget(self._divider())

        # Status
        self._status_lbl = QLabel()
        self._status_lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color: #585b70; font-size: 11px; padding: 4px 2px;")
        layout.addWidget(self._status_lbl)
        layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Activate default tool
        self._tool_btns['rect'].setChecked(True)
        return scroll

    # ── widget helpers ────────────────────────────────────────────────────────

    def _divider(self):
        line = QFrame(); line.setObjectName("divider")
        line.setFrameShape(QFrame.Shape.HLine); return line

    def _section_label(self, text):
        lbl = QLabel(text); lbl.setObjectName("section_lbl")
        lbl.setContentsMargins(2, 6, 2, 2); return lbl

    def _tool_button(self, icon, label, shortcut):
        btn = QPushButton(f"{icon}  {label}")
        btn.setObjectName("tool_btn"); btn.setCheckable(True)
        btn.setAutoExclusive(True)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setToolTip(f"{label}  [{shortcut}]"); return btn

    def _action_button(self, icon, label, shortcut):
        btn = QPushButton(f"{icon}  {label}")
        btn.setObjectName("action_btn")
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setToolTip(f"{label}  [{shortcut}]"); return btn

    def _refresh_below_color_btn(self):
        r, g, b = self.rm.mask_below_color
        luma = 0.299*r + 0.587*g + 0.114*b
        fg   = "#000" if luma > 128 else "#fff"
        self._below_color_btn.setStyleSheet(
            f"QPushButton#action_btn {{ background-color: rgb({r},{g},{b}); "
            f"color: {fg}; border: 1px solid #313244; border-radius: 8px; "
            "padding: 8px 10px; text-align: left; }"
            f"QPushButton#action_btn:hover {{ background-color: rgb({min(r+20,255)},{min(g+20,255)},{min(b+20,255)}); }}"
        )

    def _refresh_above_color_btn(self):
        r, g, b = self.rm.mask_above_color
        luma = 0.299*r + 0.587*g + 0.114*b
        fg   = "#000" if luma > 128 else "#fff"
        self._above_color_btn.setStyleSheet(
            f"QPushButton#action_btn {{ background-color: rgb({r},{g},{b}); "
            f"color: {fg}; border: 1px solid #313244; border-radius: 8px; "
            "padding: 8px 10px; text-align: left; }"
            f"QPushButton#action_btn:hover {{ background-color: rgb({min(r+20,255)},{min(g+20,255)},{min(b+20,255)}); }}"
        )

    # ── actions ───────────────────────────────────────────────────────────────

    def _on_tool_toggled(self, key, on):
        if on:
            self.canvas.mode = key
            self.canvas.selected_idx = -1
            self.canvas.update()
            self._refresh_status()

    def _on_mask_type_btn(self, btn):
        labels = {'◻  Binary': 'binary', '◼  Multi-ID': 'multi', '⊞  Both': 'both'}
        self.rm.mask_type = labels.get(btn.text(), 'multi')
        self._refresh_status()

    def _toggle_bg(self):
        self.rm.show_bg = not self.rm.show_bg
        self._bg_btn.setStyleSheet(
            "" if self.rm.show_bg else
            "QPushButton#action_btn { background-color: #1e66f5; color: #fff; "
            "border: 1px solid #1d62e8; border-radius: 8px; padding: 8px 10px; }"
        )
        self.canvas.update()

    def _delete_selected(self):
        self.canvas._delete_selected()

    def _on_intensity_toggled(self, active):
        self.rm.intensity_active = active
        self.canvas.update_intensity_overlay()
        self.canvas.update()
        self._refresh_status()

    def _on_lo_changed(self, val):
        val = min(val, self.rm.intensity_high)
        self._lo_slider.blockSignals(True)
        self._lo_slider.setValue(val)
        self._lo_slider.blockSignals(False)
        self._lo_spin.blockSignals(True)
        self._lo_spin.setValue(val)
        self._lo_spin.blockSignals(False)
        self.rm.intensity_low = val
        self.canvas.update_intensity_overlay()
        self.canvas.update()

    def _on_lo_spin_changed(self, val):
        self._lo_slider.setValue(val)

    def _on_hi_changed(self, val):
        val = max(val, self.rm.intensity_low)
        self._hi_slider.blockSignals(True)
        self._hi_slider.setValue(val)
        self._hi_slider.blockSignals(False)
        self._hi_spin.blockSignals(True)
        self._hi_spin.setValue(val)
        self._hi_spin.blockSignals(False)
        self.rm.intensity_high = val
        self.canvas.update_intensity_overlay()
        self.canvas.update()

    def _on_hi_spin_changed(self, val):
        self._hi_slider.setValue(val)

    def _pick_below_color(self):
        r, g, b = self.rm.mask_below_color
        chosen = QColorDialog.getColor(QColor(r, g, b), self, "Pick color for below-threshold pixels")
        if chosen.isValid():
            self.rm.mask_below_color = (chosen.red(), chosen.green(), chosen.blue())
            self._refresh_below_color_btn()
            self.canvas.update_intensity_overlay()
            self.canvas.update()

    def _pick_above_color(self):
        r, g, b = self.rm.mask_above_color
        chosen = QColorDialog.getColor(QColor(r, g, b), self, "Pick color for above-threshold pixels")
        if chosen.isValid():
            self.rm.mask_above_color = (chosen.red(), chosen.green(), chosen.blue())
            self._refresh_above_color_btn()
            self.canvas.update_intensity_overlay()
            self.canvas.update()

    def _create_threshold_rois(self):
        """Convert the current intensity threshold mask into ROIObject(s).

        Each contiguous region in the threshold mask becomes an unassigned ROI,
        identical in behaviour to any hand-drawn ROI.  The user then assigns IDs
        and saves them through the normal pipeline.
        """
        if not self.rm.intensity_active:
            self.statusBar().showMessage("  Enable the intensity filter first.", 3000)
            return
        n = self.rm.create_rois_from_threshold()
        if n == 0:
            self.statusBar().showMessage("  No regions found in threshold mask (try wider range).", 3000)
        else:
            self.statusBar().showMessage(
                f"  Added {n} threshold-based ROI(s).  Assign IDs before saving.", 4000)
        self.canvas.update()
        self._refresh_status()

    def _reset_ids(self):
        """Clear all ID assignments — every ROI returns to unassigned state."""
        for roi in self.rm.rois:
            roi.roi_id   = 0
            roi.assigned = False
        self.rm.assign_counter = 1
        self.canvas.update()
        self._refresh_status()

    def _delete_all(self):
        """Remove every ROI and reset the assign counter."""
        if not self.rm.rois:
            self.statusBar().showMessage("  No ROIs to delete.", 2000)
            return
        n = len(self.rm.rois)
        self.rm.rois.clear()
        self.rm.assign_counter = 1
        self.canvas.selected_idx = -1
        self.canvas.update()
        self._refresh_status()
        self.statusBar().showMessage(f"  Deleted {n} ROI(s).", 2000)

    def _auto_assign_ids(self):
        """Assign sequential IDs to all unassigned ROIs without showing a dialog."""
        pending = [r for r in self.rm.rois if not r.assigned]
        if not pending:
            self.statusBar().showMessage("  All ROIs already have IDs.", 2000)
            return
        for roi in pending:
            roi.roi_id   = self.rm.assign_counter
            roi.assigned = True
            self.rm.assign_counter += 1
        self.canvas.update()
        self._refresh_status()
        self.statusBar().showMessage(f"  Auto-assigned IDs to {len(pending)} ROI(s).", 2000)

    def _show_id_dialog(self) -> bool:
        """Show ID-assignment dialog for remaining unassigned ROIs before save."""
        dlg = IDAssignDialog(self.rm.rois, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            for i, new_id in dlg.get_assignments().items():
                self.rm.rois[i].roi_id   = new_id
                self.rm.rois[i].assigned = True
            assigned = [r.roi_id for r in self.rm.rois if r.assigned]
            self.rm.assign_counter = (max(assigned) + 1) if assigned else 1
            self.canvas.update()
            return True
        return False

    def _save_close(self):
        if self.rm.mask_type in ('multi', 'both') and self.rm.rois:
            unassigned = [r for r in self.rm.rois if not r.assigned]
            if unassigned:
                # Only show dialog when some ROIs still lack IDs
                if not self._show_id_dialog():
                    return
        self.rm.save_masks()
        self.close()

    def _cancel(self):
        self.close()

    def _refresh_status(self):
        n          = len(self.rm.rois)
        n_assigned = sum(1 for r in self.rm.rois if r.assigned)
        n_pending  = n - n_assigned
        sel        = self.canvas.selected_idx
        sel_str    = f"#{self.rm.rois[sel].roi_id}" if 0 <= sel < n else "—"
        mode_map = {
            'select': 'Select/Move', 'rect': 'Rectangle',
            'circle': 'Circle', 'freehand': 'Freehand',
            'assign': 'Assign IDs',
        }
        type_map = {'binary': 'Binary', 'multi': 'Multi-ID', 'both': 'Both'}
        hint = ("Click ROIs to number them" if self.canvas.mode == 'assign'
                else "Drag handles to resize")
        self._status_lbl.setText(
            f"Mode:    {mode_map.get(self.canvas.mode,'')}\n"
            f"Output:  {type_map.get(self.rm.mask_type,'')}\n"
            f"Filter:  {'ON' if self.rm.intensity_active else 'off'}  "
            f"[{self.rm.intensity_low}–{self.rm.intensity_high}]\n"
            f"ROIs:    {n}  (✓{n_assigned} ?{n_pending})\n"
            f"Sel:     {sel_str}"
        )
        self.statusBar().showMessage(
            f"  {mode_map.get(self.canvas.mode,'')}   │   "
            f"Output: {type_map.get(self.rm.mask_type,'')}   │   "
            f"ROIs: {n}  ✓{n_assigned} assigned  ?{n_pending} pending   │   "
            f"{hint}  ·  [R/C/F/S/A/B/Del/↵/Esc]"
        )

    # ── keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, e):
        key_map = {
            Qt.Key.Key_S: 'select', Qt.Key.Key_R: 'rect',
            Qt.Key.Key_C: 'circle', Qt.Key.Key_F: 'freehand',
            Qt.Key.Key_A: 'assign',
        }
        if e.key() in key_map:
            self._tool_btns[key_map[e.key()]].setChecked(True)
        elif e.key() == Qt.Key.Key_B:
            self._toggle_bg()
        elif e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._delete_selected()
        elif e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._save_close()
        elif e.key() == Qt.Key.Key_Escape:
            self._cancel()
        else:
            super().keyPressEvent(e)


# ─────────────────────────────────────────────────────────────────────────────
# Public ROIMaker
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Per-ROI colour palette
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    QColor( 30, 102, 245), QColor( 64, 160,  43), QColor(210,  15,  57),
    QColor(136,  57, 239), QColor(223, 142,  29), QColor(254, 100,  11),
    QColor( 23, 146, 153), QColor(156, 160, 176),
]

def _roi_color(roi_id: int) -> QColor:
    return _PALETTE[(roi_id - 1) % len(_PALETTE)]


# ─────────────────────────────────────────────────────────────────────────────
# ID Assignment Dialog (shown before saving in multi / both mode)
# ─────────────────────────────────────────────────────────────────────────────

class IDAssignDialog(QDialog):
    """Let the user rename/reorder ROI IDs before the final save."""

    def __init__(self, rois: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Region IDs")
        self.setMinimumWidth(360)
        self.setStyleSheet(_STYLE)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        lbl = QLabel("Set the ID for each drawn region.\nIDs must be positive integers (duplicates are allowed).")
        lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        self._table = QTableWidget(len(rois), 3)
        self._table.setHorizontalHeaderLabels(["Color", "Auto ID", "New ID"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.setColumnWidth(0, 28)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)

        # Next sequential number for unassigned ROIs
        _next = max((r.roi_id for r in rois if r.assigned), default=0) + 1

        self._spinboxes = []
        for row, roi in enumerate(rois):
            # colour swatch — grey for unassigned, palette colour for assigned
            c = _roi_color(roi.roi_id) if roi.assigned else QColor(120, 120, 130)
            swatch = QWidget()
            swatch.setStyleSheet(
                f"background-color: rgb({c.red()},{c.green()},{c.blue()});"
                "border-radius: 3px; margin: 4px;"
            )
            self._table.setCellWidget(row, 0, swatch)

            # status column: shows current ID or "unassigned"
            status_text = str(roi.roi_id) if roi.assigned else "unassigned"
            id_item = QTableWidgetItem(status_text)
            id_item.setTextAlignment(Qt.AlignCenter)
            id_item.setForeground(QColor("#585b70") if roi.assigned else QColor("#f38ba8"))
            self._table.setItem(row, 1, id_item)

            # editable new ID — pre-fill with assigned ID or next sequential
            default_id = roi.roi_id if roi.assigned else _next
            if not roi.assigned:
                _next += 1

            spin = QSpinBox()
            spin.setRange(1, 9999)
            spin.setValue(default_id)
            spin.setAlignment(Qt.AlignCenter)
            self._table.setCellWidget(row, 2, spin)
            self._spinboxes.append(spin)

        layout.addWidget(self._table)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_assignments(self) -> dict:
        """Return {row_index: new_id} so callers can update roi.roi_id."""
        return {i: spin.value() for i, spin in enumerate(self._spinboxes)}


# ─────────────────────────────────────────────────────────────────────────────
# Drawing canvas
# ─────────────────────────────────────────────────────────────────────────────

_HANDLE_R = 6
_MIN_BOX  = 6


class ImageCanvas(QWidget):
    roi_changed = Signal()

    def __init__(self, rm, parent=None):
        super().__init__(parent)
        self.rm = rm
        self.selected_idx = -1
        self.mode = 'rect'

        # drawing state
        self._drawing  = False
        self._start_i  = None
        self._free_i   = []

        # move state
        self._moving       = False
        self._mv_start_mi  = None
        self._mv_start_pts = None

        # handle-resize state
        self._resizing      = False
        self._rz_handle     = -1
        self._rz_start_mw   = None
        self._rz_start_bbox = None
        self._rz_start_pts  = None

        self._cur_mw = QPointF(0, 0)

        # transform
        self._scale    = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0

        # zoom / pan
        self._zoom       = 1.0
        self._pan_x      = 0.0
        self._pan_y      = 0.0
        self._fit_scale  = 1.0
        self._panning    = False
        self._pan_start_w  = None
        self._pan_start_ox = None
        self._pan_start_oy = None

        # base image
        arr = np.asarray(rm.display_base, dtype=np.uint8)
        h, w = arr.shape
        self._pixmap = QPixmap.fromImage(
            QImage(arr.tobytes(), w, h, w, QImage.Format_Grayscale8)
        )

        # intensity overlay: stored as a numpy RGBA array so paintEvent can
        # wrap it in a fresh QImage each frame without a copy or GC hazard.
        self._int_overlay = None   # np.ndarray (H, W, 4) uint8 or None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)

    # ── transform ─────────────────────────────────────────────────────────────

    def _recompute_transform(self):
        cw, ch = self.width(), self.height()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        self._fit_scale = min(cw / iw, ch / ih)
        self._scale    = self._fit_scale * self._zoom
        self._offset_x = (cw - iw * self._scale) / 2 + self._pan_x
        self._offset_y = (ch - ih * self._scale) / 2 + self._pan_y

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        if delta == 0:
            return
        self._recompute_transform()
        mx_w = e.position().x()
        my_w = e.position().y()
        # Image-space coords under the cursor before zooming
        ix, iy = self._w2i(mx_w, my_w)
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        new_zoom = max(1.0, min(self._zoom * factor, 20.0))
        if abs(new_zoom - self._zoom) < 1e-9:
            return
        self._zoom = new_zoom
        # Recompute natural center offset at new scale, then shift pan so the
        # image-space point under the cursor stays under the cursor.
        new_scale = self._fit_scale * new_zoom
        cw, ch = self.width(), self.height()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        desired_ox = mx_w - ix * new_scale
        desired_oy = my_w - iy * new_scale
        self._pan_x = desired_ox - (cw - iw * new_scale) / 2
        self._pan_y = desired_oy - (ch - ih * new_scale) / 2
        self.update()

    def reset_zoom(self):
        self._zoom  = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def _w2i(self, wx, wy):
        return ((wx - self._offset_x) / self._scale,
                (wy - self._offset_y) / self._scale)

    def _i2w(self, ix, iy):
        return (ix * self._scale + self._offset_x,
                iy * self._scale + self._offset_y)

    def _qpt_w2i(self, q):
        x, y = self._w2i(q.x(), q.y())
        return QPointF(x, y)

    def _pts_to_poly_w(self, pts):
        return QPolygonF([
            QPointF(*self._i2w(float(p[0]), float(p[1]))) for p in pts
        ])

    # ── intensity overlay ──────────────────────────────────────────────────────

    def update_intensity_overlay(self):
        """Recompute the RGBA overlay array for out-of-range pixels.

        We store the raw numpy array rather than a QPixmap so that paintEvent
        can wrap it in a QImage each frame. This avoids the deferred-copy bug
        that occurs when QPixmap.fromImage() is called on an inline QImage.
        """
        if not self.rm.intensity_active:
            self._int_overlay = None
            return
        lo, hi  = self.rm.intensity_low, self.rm.intensity_high
        arr     = self.rm._raw_img            # (H, W) float64 — original values
        below   = arr < lo
        above   = arr > hi
        rgba = np.zeros((self.rm.H, self.rm.W, 4), dtype=np.uint8)
        rb, gb, bb = self.rm.mask_below_color
        rgba[below, 0] = rb; rgba[below, 1] = gb; rgba[below, 2] = bb; rgba[below, 3] = 190
        ra, ga, ba = self.rm.mask_above_color
        rgba[above, 0] = ra; rgba[above, 1] = ga; rgba[above, 2] = ba; rgba[above, 3] = 190
        self._int_overlay = rgba              # keep array alive for QImage wrapping

    # ── bounding-box handles ───────────────────────────────────────────────────

    @staticmethod
    def _bbox(pts):
        return (float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))

    def _handle_pos_i(self, pts):
        x0, y0, x1, y1 = self._bbox(pts)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        return [(x0,y0),(mx,y0),(x1,y0),(x1,my),(x1,y1),(mx,y1),(x0,y1),(x0,my)]

    def _hit_handle(self, wpt, roi):
        thresh2 = (_HANDLE_R + 3) ** 2
        for i, (ix, iy) in enumerate(self._handle_pos_i(roi.pts)):
            wx, wy = self._i2w(ix, iy)
            if (wpt.x()-wx)**2 + (wpt.y()-wy)**2 <= thresh2:
                return i
        return -1

    def _hit_roi(self, wpt):
        ix, iy = self._w2i(wpt.x(), wpt.y())
        for i, roi in enumerate(self.rm.rois):
            if cv2.pointPolygonTest(roi.pts.astype(np.float32),
                                    (float(ix), float(iy)), False) >= 0:
                return i
        return -1

    # ── painting ───────────────────────────────────────────────────────────────

    def paintEvent(self, _):
        self._recompute_transform()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        p.fillRect(self.rect(), QColor(18, 18, 30))

        iw, ih = self._pixmap.width(), self._pixmap.height()
        target = QRectF(self._offset_x, self._offset_y,
                        iw * self._scale, ih * self._scale)

        if self.rm.show_bg:
            p.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        # intensity mask overlay — wrap the live numpy array in a QImage each
        # frame so there is no stale-pointer or deferred-copy issue.
        if self.rm.intensity_active and self._int_overlay is not None:
            h, w = self._int_overlay.shape[:2]
            _img = QImage(self._int_overlay.data, w, h, 4 * w,
                          QImage.Format_RGBA8888)
            self._int_img_ref = _img   # prevent GC while QPainter holds it
            p.drawImage(target, _img)

        for i, roi in enumerate(self.rm.rois):
            self._paint_roi(p, roi, selected=(i == self.selected_idx))

        if self._drawing and self._start_i is not None:
            self._paint_preview(p)

        p.end()

    def _paint_roi(self, p, roi, selected):
        if not roi.assigned:
            color      = QColor(0, 220, 100) if selected else QColor(140, 140, 155)
            line_style = Qt.DashLine
            label_text = "?"
        else:
            color      = QColor(0, 220, 100) if selected else _roi_color(roi.roi_id)
            line_style = Qt.SolidLine
            label_text = f"ID:{roi.roi_id}"

        poly = self._pts_to_poly_w(roi.pts)
        fill = QColor(color.red(), color.green(), color.blue(), 40 if not roi.assigned else 55)
        p.setBrush(QBrush(fill))
        pen = QPen(color, 2.5 if selected else 1.5, line_style)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.drawPolygon(poly)
        if roi.pts.shape[0]:
            wx, wy = self._i2w(float(roi.pts[0, 0]), float(roi.pts[0, 1]))
            p.setPen(QPen(Qt.white))
            p.setFont(QFont("Segoe UI", 9, QFont.Bold))
            p.drawText(int(wx) + 4, int(wy) + 13, label_text)
        if selected:
            self._paint_handles(p, roi)

    def _paint_handles(self, p, roi):
        for ix, iy in self._handle_pos_i(roi.pts):
            wx, wy = self._i2w(ix, iy)
            r = _HANDLE_R
            p.setBrush(QBrush(QColor(255, 255, 255, 230)))
            p.setPen(QPen(QColor(30, 100, 220), 1.5))
            p.drawEllipse(QRectF(wx-r, wy-r, r*2, r*2))

    def _paint_preview(self, p):
        pts = self._preview_pts()
        if len(pts) < 3:
            return
        poly = QPolygonF([QPointF(*self._i2w(float(pt[0]), float(pt[1]))) for pt in pts])
        p.setBrush(QBrush(QColor(255, 255, 255, 25)))
        p.setPen(QPen(QColor(255, 255, 255, 160), 1.2, Qt.DashLine))
        p.drawPolygon(poly)

    def _preview_pts(self):
        if self._start_i is None:
            return []
        ix, iy = self._start_i
        mx, my = self._w2i(self._cur_mw.x(), self._cur_mw.y())
        if self.mode == 'rect':
            return [[ix, iy], [mx, iy], [mx, my], [ix, my]]
        if self.mode == 'circle':
            r = max(int(np.hypot(mx-ix, my-iy)), 3)
            return cv2.ellipse2Poly((int(ix), int(iy)), (r, r), 0, 0, 360, 8).tolist()
        return list(self._free_i)

    # ── mouse ──────────────────────────────────────────────────────────────────

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._recompute_transform()
            self._panning    = True
            self._pan_start_w  = QPointF(e.pos())
            self._pan_start_ox = self._pan_x
            self._pan_start_oy = self._pan_y
            self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() != Qt.LeftButton:
            return
        wpt = QPointF(e.pos())

        # ── Assign mode: click a ROI to give it the next sequential ID ──────
        if self.mode == 'assign':
            hit = self._hit_roi(wpt)
            if hit != -1:
                roi = self.rm.rois[hit]
                if not roi.assigned:
                    roi.roi_id   = self.rm.assign_counter
                    roi.assigned = True
                    self.rm.assign_counter += 1
                self.selected_idx = hit
            else:
                self.selected_idx = -1
            self.update()
            self.roi_changed.emit()
            return
        # ────────────────────────────────────────────────────────────────────

        if self.selected_idx != -1:
            hi = self._hit_handle(wpt, self.rm.rois[self.selected_idx])
            if hi != -1:
                roi = self.rm.rois[self.selected_idx]
                self._resizing      = True
                self._rz_handle     = hi
                self._rz_start_mw   = wpt
                self._rz_start_bbox = self._bbox(roi.pts)
                self._rz_start_pts  = roi.pts.copy().astype(float)
                self.update()
                return

        hit = self._hit_roi(wpt)
        if hit != -1:
            self.selected_idx  = hit
            self._moving       = True
            self._mv_start_mi  = self._qpt_w2i(wpt)
            self._mv_start_pts = self.rm.rois[hit].pts.copy().astype(float)
            self.update(); self.roi_changed.emit(); return

        if self.mode != 'select':
            self.selected_idx = -1
            self._drawing  = True
            ix, iy         = self._w2i(wpt.x(), wpt.y())
            self._start_i  = (ix, iy)
            self._free_i   = [(ix, iy)]
        else:
            self.selected_idx = -1
        self.update(); self.roi_changed.emit()

    def mouseMoveEvent(self, e):
        wpt = QPointF(e.pos())
        self._cur_mw = wpt

        if self._panning:
            self._pan_x = self._pan_start_ox + wpt.x() - self._pan_start_w.x()
            self._pan_y = self._pan_start_oy + wpt.y() - self._pan_start_w.y()
            self.update()
            return

        if self._resizing and self.selected_idx != -1:
            self._apply_resize(wpt)
        elif self._moving and self.selected_idx != -1:
            cur_i = self._qpt_w2i(wpt)
            dx = cur_i.x() - self._mv_start_mi.x()
            dy = cur_i.y() - self._mv_start_mi.y()
            roi = self.rm.rois[self.selected_idx]
            new_pts = self._mv_start_pts + np.array([dx, dy])
            roi.pts    = np.clip(new_pts, 0, None).astype(np.int32)
            roi.center = np.mean(roi.pts, axis=0)
        elif self._drawing and self.mode == 'freehand' and self._start_i:
            ix, iy = self._w2i(wpt.x(), wpt.y())
            self._free_i.append((ix, iy))

        # adaptive cursor
        if self.selected_idx != -1 and not self._drawing:
            hi = self._hit_handle(wpt, self.rm.rois[self.selected_idx])
            cursors = {0: Qt.SizeFDiagCursor, 4: Qt.SizeFDiagCursor,
                       2: Qt.SizeBDiagCursor, 6: Qt.SizeBDiagCursor,
                       1: Qt.SizeVerCursor,   5: Qt.SizeVerCursor,
                       3: Qt.SizeHorCursor,   7: Qt.SizeHorCursor}
            if hi in cursors:
                self.setCursor(cursors[hi])
            elif self._hit_roi(wpt) != -1:
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.CrossCursor if self.mode != 'select' else Qt.ArrowCursor)
        else:
            self.setCursor(Qt.CrossCursor if self.mode not in ('select', None) else Qt.ArrowCursor)

        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        if e.button() != Qt.LeftButton:
            return
        if self._resizing:
            self._resizing = False; self.roi_changed.emit()
        if self._moving:
            self._moving = False; self.roi_changed.emit()
        if self._drawing:
            self._drawing = False
            wpt = QPointF(e.pos())
            ix, iy = self._w2i(wpt.x(), wpt.y())
            pts = self._finalise_pts(ix, iy)
            if len(pts) >= 3:
                roi = ROIObject(pts)      # roi_id=0, assigned=False until user assigns
                self.rm.rois.append(roi)
                self.selected_idx = len(self.rm.rois) - 1
            self._start_i = None; self._free_i = []
            self.roi_changed.emit()
        self.update()

    def _finalise_pts(self, mx, my):
        if self._start_i is None:
            return []
        ix, iy = self._start_i
        if self.mode == 'rect':
            return np.array([[ix,iy],[mx,iy],[mx,my],[ix,my]])
        if self.mode == 'circle':
            r = max(int(np.hypot(mx-ix, my-iy)), 3)
            return cv2.ellipse2Poly((int(ix),int(iy)), (r,r), 0, 0, 360, 8)
        pts = np.array(self._free_i)
        return pts if len(pts) >= 3 else []

    def _apply_resize(self, wpt):
        roi = self.rm.rois[self.selected_idx]
        dix = (wpt.x() - self._rz_start_mw.x()) / self._scale
        diy = (wpt.y() - self._rz_start_mw.y()) / self._scale
        x0, y0, x1, y1 = self._rz_start_bbox
        nx0, ny0, nx1, ny1 = x0, y0, x1, y1
        hi = self._rz_handle
        if hi in (0, 6, 7): nx0 = x0 + dix
        if hi in (2, 3, 4): nx1 = x1 + dix
        if hi in (0, 1, 2): ny0 = y0 + diy
        if hi in (4, 5, 6): ny1 = y1 + diy
        if nx1 - nx0 < _MIN_BOX: nx1 = nx0 + _MIN_BOX
        if ny1 - ny0 < _MIN_BOX: ny1 = ny0 + _MIN_BOX
        bw = max(x1 - x0, 1.0); bh = max(y1 - y0, 1.0)
        pts = self._rz_start_pts.copy()
        pts[:, 0] = nx0 + (pts[:, 0] - x0) * (nx1 - nx0) / bw
        pts[:, 1] = ny0 + (pts[:, 1] - y0) * (ny1 - ny0) / bh
        roi.pts    = np.clip(pts, 0, None).astype(np.int32)
        roi.center = np.mean(roi.pts, axis=0)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._delete_selected()
        elif e.key() == Qt.Key_Home:
            self.reset_zoom()

    def _delete_selected(self):
        if 0 <= self.selected_idx < len(self.rm.rois):
            self.rm.rois.pop(self.selected_idx)
            self.selected_idx = -1
            # Recalculate assign_counter so it stays above all existing IDs
            assigned_ids = [r.roi_id for r in self.rm.rois if r.assigned]
            self.rm.assign_counter = (max(assigned_ids) + 1) if assigned_ids else 1
            self.roi_changed.emit(); self.update()


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class ROIApp(QMainWindow):
    def __init__(self, rm):
        super().__init__()
        self.rm = rm
        self.setWindowTitle("ROI Maker")
        self.setStyleSheet(_STYLE)

        central     = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Canvas first so _build_sidebar signals can reference it safely
        self.canvas = ImageCanvas(rm, self)
        self.canvas.roi_changed.connect(self._refresh_status)
        self.canvas.update_intensity_overlay()

        sidebar = self._build_sidebar()
        root_layout.addWidget(sidebar)
        root_layout.addWidget(self.canvas, stretch=1)

        self.setStatusBar(QStatusBar(self))
        self._refresh_status()

    # ── sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: #181825; }"
                             "QScrollBar:vertical { width: 6px; background: #181825; }"
                             "QScrollBar::handle:vertical { background: #313244; border-radius: 3px; }")
        scroll.setFixedWidth(185)

        sb = QWidget()
        sb.setObjectName("sidebar")
        layout = QVBoxLayout(sb)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(4)
        scroll.setWidget(sb)

        # Title
        title = QLabel("⬡  ROI Maker")
        title.setObjectName("title_lbl")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(self._divider())

        # ── TOOLS ──
        layout.addWidget(self._section_label("TOOLS"))
        self._tool_btns = {}
        for key, icon, label, sc in [
            ('select',   '↖', 'Select / Move', 'S'),
            ('rect',     '▭', 'Rectangle',     'R'),
            ('circle',   '◯', 'Circle',        'C'),
            ('freehand', '✏', 'Freehand',      'F'),
            ('assign',   '⊕', 'Assign IDs',    'A'),
        ]:
            btn = self._tool_button(icon, label, sc)
            btn.toggled.connect(lambda on, k=key: self._on_tool_toggled(k, on))
            self._tool_btns[key] = btn
            layout.addWidget(btn)

        reset_btn = self._action_button('↺', 'Reset IDs', '')
        reset_btn.setToolTip("Clear all ID assignments — ROIs return to unassigned state")
        reset_btn.clicked.connect(self._reset_ids)
        layout.addWidget(reset_btn)

        auto_btn = self._action_button('⟳', 'Auto-assign IDs', '')
        auto_btn.setToolTip("Assign sequential IDs to all unassigned ROIs automatically")
        auto_btn.clicked.connect(self._auto_assign_ids)
        layout.addWidget(auto_btn)

        layout.addSpacing(4); layout.addWidget(self._divider())

        # ── OUTPUT TYPE ──
        layout.addWidget(self._section_label("OUTPUT TYPE"))
        self._mask_type_group = QButtonGroup(self)
        self._mask_type_group.setExclusive(True)
        for val, icon, label in [
            ('binary', '◻', 'Binary'),
            ('multi',  '◼', 'Multi-ID'),
            ('both',   '⊞', 'Both'),
        ]:
            btn = QPushButton(f"{icon}  {label}")
            btn.setObjectName("tool_btn")
            btn.setCheckable(True)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.setToolTip(f"Save as: {label}")
            self._mask_type_group.addButton(btn)
            layout.addWidget(btn)
            if val == self.rm.mask_type:
                btn.setChecked(True)
        self._mask_type_group.buttonClicked.connect(self._on_mask_type_btn)
        # store reference to buttons by order
        self._mask_type_btns = {
            v: self._mask_type_group.buttons()[i]
            for i, v in enumerate(['binary', 'multi', 'both'])
        }

        layout.addSpacing(4); layout.addWidget(self._divider())

        # ── EDIT ──
        layout.addWidget(self._section_label("EDIT"))
        self._bg_btn = self._action_button('◑', 'Toggle Image', 'B')
        self._bg_btn.clicked.connect(self._toggle_bg)
        layout.addWidget(self._bg_btn)

        del_btn = self._action_button('✕', 'Delete ROI', 'Del')
        del_btn.clicked.connect(self._delete_selected)
        layout.addWidget(del_btn)

        del_all_btn = self._action_button('⊘', 'Delete All ROIs', '')
        del_all_btn.setToolTip("Remove every ROI and start fresh")
        del_all_btn.clicked.connect(self._delete_all)
        layout.addWidget(del_all_btn)

        layout.addSpacing(4); layout.addWidget(self._divider())

        # ── INTENSITY FILTER ──
        layout.addWidget(self._section_label("INTENSITY FILTER"))

        self._int_btn = QPushButton("⚡  Enable Filter")
        self._int_btn.setObjectName("action_btn")
        self._int_btn.setCheckable(True)
        self._int_btn.setChecked(self.rm.intensity_active)
        self._int_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._int_btn.toggled.connect(self._on_intensity_toggled)
        layout.addWidget(self._int_btn)

        # Low threshold
        self._lo_lbl = QLabel("Low threshold")
        self._lo_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(self._lo_lbl)
        _lo_row = QWidget(); _lo_rl = QHBoxLayout(_lo_row)
        _lo_rl.setContentsMargins(0, 0, 0, 0); _lo_rl.setSpacing(4)
        self._lo_slider = QSlider(Qt.Horizontal)
        self._lo_slider.setRange(self.rm.img_min, self.rm.img_max)
        self._lo_slider.setValue(self.rm.intensity_low)
        self._lo_slider.valueChanged.connect(self._on_lo_changed)
        _lo_rl.addWidget(self._lo_slider)
        self._lo_spin = QSpinBox()
        self._lo_spin.setRange(self.rm.img_min, self.rm.img_max)
        self._lo_spin.setValue(self.rm.intensity_low)
        self._lo_spin.setFixedWidth(58)
        self._lo_spin.valueChanged.connect(self._on_lo_spin_changed)
        _lo_rl.addWidget(self._lo_spin)
        layout.addWidget(_lo_row)

        # High threshold
        self._hi_lbl = QLabel("High threshold")
        self._hi_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(self._hi_lbl)
        _hi_row = QWidget(); _hi_rl = QHBoxLayout(_hi_row)
        _hi_rl.setContentsMargins(0, 0, 0, 0); _hi_rl.setSpacing(4)
        self._hi_slider = QSlider(Qt.Horizontal)
        self._hi_slider.setRange(self.rm.img_min, self.rm.img_max)
        self._hi_slider.setValue(self.rm.intensity_high)
        self._hi_slider.valueChanged.connect(self._on_hi_changed)
        _hi_rl.addWidget(self._hi_slider)
        self._hi_spin = QSpinBox()
        self._hi_spin.setRange(self.rm.img_min, self.rm.img_max)
        self._hi_spin.setValue(self.rm.intensity_high)
        self._hi_spin.setFixedWidth(58)
        self._hi_spin.valueChanged.connect(self._on_hi_spin_changed)
        _hi_rl.addWidget(self._hi_spin)
        layout.addWidget(_hi_row)

        # Mask-color pickers (separate for below/above)
        self._below_color_btn = QPushButton("▼  Below color")
        self._below_color_btn.setObjectName("action_btn")
        self._below_color_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._below_color_btn.setToolTip("Color for pixels below low threshold")
        self._below_color_btn.clicked.connect(self._pick_below_color)
        self._refresh_below_color_btn()
        layout.addWidget(self._below_color_btn)

        self._above_color_btn = QPushButton("▲  Above color")
        self._above_color_btn.setObjectName("action_btn")
        self._above_color_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._above_color_btn.setToolTip("Color for pixels above high threshold")
        self._above_color_btn.clicked.connect(self._pick_above_color)
        self._refresh_above_color_btn()
        layout.addWidget(self._above_color_btn)

        thresh_btn = self._action_button('⊞', 'Create Threshold ROI', '')
        thresh_btn.setToolTip(
            "Convert the current intensity threshold into ROI(s).\n"
            "They are added as unassigned ROIs — assign IDs the same way as drawn ROIs."
        )
        thresh_btn.clicked.connect(self._create_threshold_rois)
        layout.addWidget(thresh_btn)

        layout.addSpacing(4); layout.addWidget(self._divider())

        # ── FILE ──
        layout.addWidget(self._section_label("FILE"))

        save_btn = QPushButton("✓  Save && Close  ↵")
        save_btn.setObjectName("save_btn")
        save_btn.setCursor(QCursor(Qt.PointingHandCursor))
        save_btn.clicked.connect(self._save_close)
        layout.addWidget(save_btn)

        cancel_btn = QPushButton("✗  Cancel  Esc")
        cancel_btn.setObjectName("cancel_btn")
        cancel_btn.setCursor(QCursor(Qt.PointingHandCursor))
        cancel_btn.clicked.connect(self._cancel)
        layout.addWidget(cancel_btn)

        layout.addSpacing(4); layout.addWidget(self._divider())

        # Status
        self._status_lbl = QLabel()
        self._status_lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color: #585b70; font-size: 11px; padding: 4px 2px;")
        layout.addWidget(self._status_lbl)
        layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Activate default tool
        self._tool_btns['rect'].setChecked(True)
        return scroll

    # ── widget helpers ────────────────────────────────────────────────────────

    def _divider(self):
        line = QFrame(); line.setObjectName("divider")
        line.setFrameShape(QFrame.HLine); return line

    def _section_label(self, text):
        lbl = QLabel(text); lbl.setObjectName("section_lbl")
        lbl.setContentsMargins(2, 6, 2, 2); return lbl

    def _tool_button(self, icon, label, shortcut):
        btn = QPushButton(f"{icon}  {label}")
        btn.setObjectName("tool_btn"); btn.setCheckable(True)
        btn.setAutoExclusive(True); btn.setCursor(QCursor(Qt.PointingHandCursor))
        btn.setToolTip(f"{label}  [{shortcut}]"); return btn

    def _action_button(self, icon, label, shortcut):
        btn = QPushButton(f"{icon}  {label}")
        btn.setObjectName("action_btn"); btn.setCursor(QCursor(Qt.PointingHandCursor))
        btn.setToolTip(f"{label}  [{shortcut}]"); return btn

    def _refresh_below_color_btn(self):
        r, g, b = self.rm.mask_below_color
        luma = 0.299*r + 0.587*g + 0.114*b
        fg   = "#000" if luma > 128 else "#fff"
        self._below_color_btn.setStyleSheet(
            f"QPushButton#action_btn {{ background-color: rgb({r},{g},{b}); "
            f"color: {fg}; border: 1px solid #313244; border-radius: 8px; "
            "padding: 8px 10px; text-align: left; }"
            f"QPushButton#action_btn:hover {{ background-color: rgb({min(r+20,255)},{min(g+20,255)},{min(b+20,255)}); }}"
        )

    def _refresh_above_color_btn(self):
        r, g, b = self.rm.mask_above_color
        luma = 0.299*r + 0.587*g + 0.114*b
        fg   = "#000" if luma > 128 else "#fff"
        self._above_color_btn.setStyleSheet(
            f"QPushButton#action_btn {{ background-color: rgb({r},{g},{b}); "
            f"color: {fg}; border: 1px solid #313244; border-radius: 8px; "
            "padding: 8px 10px; text-align: left; }"
            f"QPushButton#action_btn:hover {{ background-color: rgb({min(r+20,255)},{min(g+20,255)},{min(b+20,255)}); }}"
        )

    # ── actions ───────────────────────────────────────────────────────────────

    def _on_tool_toggled(self, key, on):
        if on:
            self.canvas.mode = key
            self.canvas.selected_idx = -1
            self.canvas.update()
            self._refresh_status()

    def _on_mask_type_btn(self, btn):
        labels = {'◻  Binary': 'binary', '◼  Multi-ID': 'multi', '⊞  Both': 'both'}
        self.rm.mask_type = labels.get(btn.text(), 'multi')
        self._refresh_status()

    def _toggle_bg(self):
        self.rm.show_bg = not self.rm.show_bg
        self._bg_btn.setStyleSheet(
            "" if self.rm.show_bg else
            "QPushButton#action_btn { background-color: #1e66f5; color: #fff; "
            "border: 1px solid #1d62e8; border-radius: 8px; padding: 8px 10px; }"
        )
        self.canvas.update()

    def _delete_selected(self):
        self.canvas._delete_selected()

    def _on_intensity_toggled(self, active):
        self.rm.intensity_active = active
        self.canvas.update_intensity_overlay()
        self.canvas.update()
        self._refresh_status()

    def _on_lo_changed(self, val):
        val = min(val, self.rm.intensity_high)
        self._lo_slider.blockSignals(True)
        self._lo_slider.setValue(val)
        self._lo_slider.blockSignals(False)
        self._lo_spin.blockSignals(True)
        self._lo_spin.setValue(val)
        self._lo_spin.blockSignals(False)
        self.rm.intensity_low = val
        self.canvas.update_intensity_overlay()
        self.canvas.update()

    def _on_lo_spin_changed(self, val):
        self._lo_slider.setValue(val)

    def _on_hi_changed(self, val):
        val = max(val, self.rm.intensity_low)
        self._hi_slider.blockSignals(True)
        self._hi_slider.setValue(val)
        self._hi_slider.blockSignals(False)
        self._hi_spin.blockSignals(True)
        self._hi_spin.setValue(val)
        self._hi_spin.blockSignals(False)
        self.rm.intensity_high = val
        self.canvas.update_intensity_overlay()
        self.canvas.update()

    def _on_hi_spin_changed(self, val):
        self._hi_slider.setValue(val)

    def _pick_below_color(self):
        r, g, b = self.rm.mask_below_color
        chosen = QColorDialog.getColor(QColor(r, g, b), self, "Pick color for below-threshold pixels")
        if chosen.isValid():
            self.rm.mask_below_color = (chosen.red(), chosen.green(), chosen.blue())
            self._refresh_below_color_btn()
            self.canvas.update_intensity_overlay()
            self.canvas.update()

    def _pick_above_color(self):
        r, g, b = self.rm.mask_above_color
        chosen = QColorDialog.getColor(QColor(r, g, b), self, "Pick color for above-threshold pixels")
        if chosen.isValid():
            self.rm.mask_above_color = (chosen.red(), chosen.green(), chosen.blue())
            self._refresh_above_color_btn()
            self.canvas.update_intensity_overlay()
            self.canvas.update()

    def _create_threshold_rois(self):
        """Convert the current intensity threshold mask into ROIObject(s).

        Each contiguous region in the threshold mask becomes an unassigned ROI,
        identical in behaviour to any hand-drawn ROI.  The user then assigns IDs
        and saves them through the normal pipeline.
        """
        if not self.rm.intensity_active:
            self.statusBar().showMessage("  Enable the intensity filter first.", 3000)
            return
        n = self.rm.create_rois_from_threshold()
        if n == 0:
            self.statusBar().showMessage("  No regions found in threshold mask (try wider range).", 3000)
        else:
            self.statusBar().showMessage(
                f"  Added {n} threshold-based ROI(s).  Assign IDs before saving.", 4000)
        self.canvas.update()
        self._refresh_status()

    def _reset_ids(self):
        """Clear all ID assignments — every ROI returns to unassigned state."""
        for roi in self.rm.rois:
            roi.roi_id   = 0
            roi.assigned = False
        self.rm.assign_counter = 1
        self.canvas.update()
        self._refresh_status()

    def _delete_all(self):
        """Remove every ROI and reset the assign counter."""
        if not self.rm.rois:
            self.statusBar().showMessage("  No ROIs to delete.", 2000)
            return
        n = len(self.rm.rois)
        self.rm.rois.clear()
        self.rm.assign_counter = 1
        self.canvas.selected_idx = -1
        self.canvas.update()
        self._refresh_status()
        self.statusBar().showMessage(f"  Deleted {n} ROI(s).", 2000)

    def _auto_assign_ids(self):
        """Assign sequential IDs to all unassigned ROIs without showing a dialog."""
        pending = [r for r in self.rm.rois if not r.assigned]
        if not pending:
            self.statusBar().showMessage("  All ROIs already have IDs.", 2000)
            return
        for roi in pending:
            roi.roi_id   = self.rm.assign_counter
            roi.assigned = True
            self.rm.assign_counter += 1
        self.canvas.update()
        self._refresh_status()
        self.statusBar().showMessage(f"  Auto-assigned IDs to {len(pending)} ROI(s).", 2000)

    def _show_id_dialog(self) -> bool:
        """Show ID-assignment dialog for remaining unassigned ROIs before save."""
        dlg = IDAssignDialog(self.rm.rois, self)
        if dlg.exec_() == QDialog.Accepted:
            for i, new_id in dlg.get_assignments().items():
                self.rm.rois[i].roi_id   = new_id
                self.rm.rois[i].assigned = True
            assigned = [r.roi_id for r in self.rm.rois if r.assigned]
            self.rm.assign_counter = (max(assigned) + 1) if assigned else 1
            self.canvas.update()
            return True
        return False

    def _save_close(self):
        if self.rm.mask_type in ('multi', 'both') and self.rm.rois:
            unassigned = [r for r in self.rm.rois if not r.assigned]
            if unassigned:
                # Only show dialog when some ROIs still lack IDs
                if not self._show_id_dialog():
                    return
        self.rm.save_masks()
        self.close()

    def _cancel(self):
        self.close()

    def _refresh_status(self):
        n          = len(self.rm.rois)
        n_assigned = sum(1 for r in self.rm.rois if r.assigned)
        n_pending  = n - n_assigned
        sel        = self.canvas.selected_idx
        sel_str    = f"#{self.rm.rois[sel].roi_id}" if 0 <= sel < n else "—"
        mode_map = {
            'select': 'Select/Move', 'rect': 'Rectangle',
            'circle': 'Circle', 'freehand': 'Freehand',
            'assign': 'Assign IDs',
        }
        type_map = {'binary': 'Binary', 'multi': 'Multi-ID', 'both': 'Both'}
        hint = ("Click ROIs to number them" if self.canvas.mode == 'assign'
                else "Drag handles to resize")
        self._status_lbl.setText(
            f"Mode:    {mode_map.get(self.canvas.mode,'')}\n"
            f"Output:  {type_map.get(self.rm.mask_type,'')}\n"
            f"Filter:  {'ON' if self.rm.intensity_active else 'off'}  "
            f"[{self.rm.intensity_low}–{self.rm.intensity_high}]\n"
            f"ROIs:    {n}  (✓{n_assigned} ?{n_pending})\n"
            f"Sel:     {sel_str}"
        )
        self.statusBar().showMessage(
            f"  {mode_map.get(self.canvas.mode,'')}   │   "
            f"Output: {type_map.get(self.rm.mask_type,'')}   │   "
            f"ROIs: {n}  ✓{n_assigned} assigned  ?{n_pending} pending   │   "
            f"{hint}  ·  [R/C/F/S/A/B/Del/↵/Esc]"
        )

    # ── keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, e):
        key_map = {Qt.Key_S:'select', Qt.Key_R:'rect',
                   Qt.Key_C:'circle', Qt.Key_F:'freehand', Qt.Key_A:'assign'}
        if e.key() in key_map:
            self._tool_btns[key_map[e.key()]].setChecked(True)
        elif e.key() == Qt.Key_B:
            self._toggle_bg()
        elif e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._delete_selected()
        elif e.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._save_close()
        elif e.key() == Qt.Key_Escape:
            self._cancel()
        elif e.key() == Qt.Key_Home:
            self.canvas.reset_zoom()
        else:
            super().keyPressEvent(e)


# ─────────────────────────────────────────────────────────────────────────────
# Public ROIMaker
# ─────────────────────────────────────────────────────────────────────────────

class ROIMaker:
    def __init__(self, image_2d, save_path="masks/mask.npy"):
        arr = np.asarray(image_2d, dtype=np.float64)
        self._raw_img       = arr                        # original values for thresholding
        self.img_min        = int(np.floor(arr.min()))
        self.img_max        = int(np.ceil(arr.max()))
        self.display_base   = cv2.normalize(arr, None, 0, 255,
                                            cv2.NORM_MINMAX).astype(np.uint8)
        self.H, self.W      = arr.shape
        self.save_path      = save_path
        self.rois           = []
        self.assign_counter = 1     # next ID to hand out in assign mode
        self.show_bg        = True

        # output type: 'binary' | 'multi' | 'both'
        self.mask_type = 'multi'

        # intensity filter — defaults span the full image range
        self.intensity_active    = False
        self.intensity_low       = self.img_min
        self.intensity_high      = self.img_max
        self.mask_below_color    = (0, 0, 139)  # RGB — pixels below low threshold
        self.mask_above_color    = (139, 0, 0)  # RGB — pixels above high threshold

        if os.path.exists(self.save_path):
            self.load_mask(self.save_path)

    # ── mask generators ───────────────────────────────────────────────────────

    def get_intensity_mask(self) -> np.ndarray:
        """Binary (H,W) uint8: 1 where pixel intensity is inside [low, high].
        Always independent of the ROI masks — saved as a separate file."""
        lo, hi = self.intensity_low, self.intensity_high
        return ((self._raw_img >= lo) & (self._raw_img <= hi)).astype(np.uint8)

    def create_rois_from_threshold(self, min_area: int = 10) -> int:
        """Convert the current intensity threshold mask into ROIObjects.

        Each contiguous region in the threshold mask is added to ``self.rois``
        as an unassigned ROI, indistinguishable from a hand-drawn one.  The
        caller is responsible for assigning IDs and saving via the normal flow.

        Parameters
        ----------
        min_area : minimum contour area in pixels (default 10).

        Returns
        -------
        Number of ROIs added.
        """
        mask = self.get_intensity_mask()          # (H, W) uint8
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        added = 0
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_area:
                continue
            roi = ROIObject(cnt.reshape(-1, 2))   # assigned=False, roi_id=0
            self.rois.append(roi)
            added += 1
        return added

    def get_binary_mask(self) -> np.ndarray:
        """All drawn ROIs → 1, background → 0.  Intensity filter NOT applied."""
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        for roi in self.rois:
            cv2.fillPoly(mask, [roi.pts.astype(np.int32)], 1)
        return mask

    def get_threshold_binary_mask(self) -> np.ndarray:
        """Binary (H,W) uint8 mask built purely from intensity thresholds.

        Every pixel whose value is within [intensity_low, intensity_high] is 1;
        everything else is 0.  No polygon filling is involved, so pixels that
        sit inside a closed/ring-shaped ROI boundary but fall outside the
        intensity range are correctly excluded — the problem that arises when
        fillPoly floods a hollow structure's interior.
        """
        lo, hi = self.intensity_low, self.intensity_high
        return ((self._raw_img >= lo) & (self._raw_img <= hi)).astype(np.uint8)

    def get_multi_cluster_mask(self) -> np.ndarray:
        """Each ROI → its roi_id; background → 0.  Intensity filter NOT applied."""
        mask = np.zeros((self.H, self.W), dtype=np.int32)
        for roi in self.rois:
            cv2.fillPoly(mask, [roi.pts.astype(np.int32)], int(roi.roi_id))
        return mask

    # ── load / save ───────────────────────────────────────────────────────────

    def load_mask(self, path: str):
        try:
            loaded = np.load(path)
            for uid in np.unique(loaded):
                if uid == 0:
                    continue
                cnts, _ = cv2.findContours((loaded == uid).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    if cv2.contourArea(cnt) > 5:
                        roi = ROIObject(cnt.reshape(-1, 2), uid)
                        roi.assigned = True   # loaded ROIs already have IDs
                        self.rois.append(roi)
            if self.rois:
                self.assign_counter = max(r.roi_id for r in self.rois) + 1
        except Exception as exc:
            print(f"Load mask failed: {exc}")

    def save_masks(self):
        stem, _ = os.path.splitext(os.path.abspath(self.save_path))
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)

        if self.mask_type == 'binary':
            if self.intensity_active:
                mask = self.get_threshold_binary_mask()
                np.save(self.save_path, mask)
                print(f"Saved threshold binary mask (intensity [{self.intensity_low}–{self.intensity_high}]) → {self.save_path}")
            else:
                np.save(self.save_path, self.get_binary_mask())
                print(f"Saved binary mask → {self.save_path}")

        elif self.mask_type == 'multi':
            m = self.get_multi_cluster_mask()
            np.save(self.save_path, m)
            print(f"Saved multi-ID mask ({len(np.unique(m))-1} region(s)) → {self.save_path}")

        elif self.mask_type == 'both':
            binary_path = f"{stem}_binary.npy"
            multi_path  = f"{stem}_multi.npy"
            np.save(binary_path, self.get_binary_mask())
            print(f"Saved binary mask       → {binary_path}")
            m = self.get_multi_cluster_mask()
            np.save(multi_path, m)
            print(f"Saved multi-ID mask ({len(np.unique(m))-1} region(s)) → {multi_path}")

        # Intensity mask is always its own separate file — never merged into ROI masks
        if self.intensity_active:
            path = f"{stem}_intensity.npy"
            np.save(path, self.get_intensity_mask())
            print(f"Saved intensity mask    → {path}")

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(self):
        """Open the editor window (blocks). Returns the chosen mask type."""
        app = QApplication.instance() or QApplication(sys.argv)
        win = ROIApp(self)
        win.resize(min(self.W + 210, 1440), min(self.H + 60, 920))
        win.show()
        app.exec()
        return self.get_multi_cluster_mask()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pyfli import DataOperations

    _HERE = os.path.dirname(os.path.abspath(__file__))

    loader = DataOperations(
        data_path=os.path.normpath(os.path.join(_HERE, "../../../../data/ICCD/mouseR_740bp")),
        irf_path =os.path.normpath(os.path.join(_HERE, "../../../../data/ICCD/mouseR_IRF")),
    )

    fli_cube = loader.load_data()
    if fli_cube is None:
        raise FileNotFoundError(f"Data not found. Tried: {os.path.abspath(loader.data_path)}")
    print(f"FLI shape: {fli_cube.shape}")

    irf_cube = loader.load_irf()
    if irf_cube is None:
        raise FileNotFoundError(f"IRF not found. Tried: {os.path.abspath(loader.irf_path)}")
    print(f"IRF shape: {irf_cube.shape}")

    intensity_proj = np.sum(fli_cube, axis=-1)
    maker = ROIMaker(intensity_proj, save_path="mouseL_mask.npy")
    multi = maker.draw()

    import matplotlib.pyplot as plt
    plt.imshow(multi); plt.show()
