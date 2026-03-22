#!/usr/bin/env python3
"""
jt-live-whisper 桌面字幕浮層

跨平台（macOS / Windows）半透明字幕浮層，透過 WebSocket 或 TCP
連線至 jt-live-whisper 即時接收辨識與翻譯結果，以浮動視窗顯示。

用法：
    python3 subtitle_overlay.py
    python3 subtitle_overlay.py --ws-url ws://192.168.1.40:19781/ws
    python3 subtitle_overlay.py --config /path/to/config.json
"""

import sys
import os
import json
import argparse
import platform
import signal

# ---------------------------------------------------------------------------
# PyQt6 guard
# ---------------------------------------------------------------------------
try:
    from PyQt6.QtCore import (
        Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QByteArray,
        pyqtProperty, QRectF, QUrl, pyqtSignal,
    )
    from PyQt6.QtGui import (
        QPainter, QColor, QFont, QBrush, QPen, QIcon, QAction, QFontMetrics,
        QCursor,
    )
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
        QPushButton, QSystemTrayIcon, QMenu, QGraphicsOpacityEffect,
    )
except ImportError:
    print("=" * 60)
    print("錯誤：找不到 PyQt6，請先安裝：")
    print("    pip install PyQt6 PyQt6-WebSockets")
    print("=" * 60)
    sys.exit(1)

try:
    from PyQt6.QtWebSockets import QWebSocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

IS_MACOS = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_WS_URL = "ws://127.0.0.1:19781/ws"
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 19780

# 使用系統預設字型（不指定特定字型名，避免不存在時 Qt 搜尋耗時）

FONT_PRESETS = {
    "small":  {"src": 13, "dst": 17},
    "medium": {"src": 16, "dst": 22},
    "large":  {"src": 20, "dst": 28},
}

IDLE_FADE_SECS = 10
IDLE_OPACITY = 0.30
RECONNECT_MS = 3000
FADE_IN_MS = 150

CONFIG_SECTION = "subtitle_overlay"


# ---------------------------------------------------------------------------
# TCP reader (fallback) — runs in a QThread-like timer loop
# ---------------------------------------------------------------------------
from PyQt6.QtNetwork import QTcpSocket, QAbstractSocket


# ---------------------------------------------------------------------------
# Main overlay widget
# ---------------------------------------------------------------------------
class SubtitleOverlay(QWidget):
    """半透明桌面字幕浮層"""

    _bg_opacity = 1.0  # internal property for animation

    def __init__(self, cfg: dict):
        super().__init__()
        self._cfg = cfg
        self._drag_pos = None
        self._hover = False
        self._click_through = cfg.get("click_through", False)
        self._single_line = cfg.get("single_line", False)
        self._font_preset = cfg.get("font_preset", "medium")
        self._bg_alpha = cfg.get("opacity", 65) / 100.0
        self._target_bg_alpha = self._bg_alpha
        self._idle = False

        # Connection
        self._ws = None
        self._tcp = None
        self._tcp_buf = b""
        self._connected = False
        self._ws_url = cfg.get("ws_url", DEFAULT_WS_URL)
        self._tcp_host = cfg.get("tcp_host", DEFAULT_TCP_HOST)
        self._tcp_port = cfg.get("tcp_port", DEFAULT_TCP_PORT)

        self._init_ui()
        self._init_tray()
        self._init_timers()
        self._restore_position()
        self._apply_click_through()
        self._connect()

    # ---- UI setup ---------------------------------------------------------

    def _init_ui(self):
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)

        # Size — 初始用單語高度，收到雙語字幕時自動變高
        self._is_bilingual = False
        screen = QApplication.primaryScreen().geometry()
        w = min(int(screen.width() * 0.7), 1000)
        h = 42
        x = (screen.width() - w) // 2
        y = screen.height() - h - 60
        self.setGeometry(x, y, w, h)
        self.setMinimumWidth(300)
        self.setMinimumHeight(40)
        self.setMouseTracking(True)  # 不按鍵也能追蹤滑鼠位置（邊緣游標變化）
        self._resize_edge = None  # 拖拉調整大小用

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 4, 16, 4)
        layout.setSpacing(0)

        # Source label
        self._src_label = QLabel("")
        self._src_label.setWordWrap(True)
        self._src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._src_label)

        # Translated label
        self._dst_label = QLabel("連線中...")
        self._dst_label.setWordWrap(True)
        self._dst_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._dst_label)

        # Close button (top-right, hidden until hover)
        self._close_btn = QPushButton("✕", self)
        self._close_btn.setFixedSize(24, 24)
        self._close_btn.setStyleSheet(
            "QPushButton { background: rgba(255,255,255,0.15); color: white;"
            " border: none; border-radius: 12px; font-size: 14px; }"
            "QPushButton:hover { background: rgba(255,80,80,0.7); }"
        )
        self._close_btn.hide()
        self._close_btn.clicked.connect(self.close)

        self._apply_fonts()

    def _apply_fonts(self):
        preset = FONT_PRESETS.get(self._font_preset, FONT_PRESETS["medium"])
        src_size = self._cfg.get("src_font_size", preset["src"])
        dst_size = self._cfg.get("dst_font_size", preset["dst"])
        font_family = self._cfg.get("font_family", "")

        src_font = QFont(font_family) if font_family else QFont()
        src_font.setPointSize(src_size)
        self._src_label.setFont(src_font)
        self._src_label.setStyleSheet("color: rgba(255,255,255,0.70); margin: 0; padding: 0;")

        dst_font = QFont(font_family) if font_family else QFont()
        dst_font.setPointSize(dst_size)
        dst_font.setBold(True)
        self._dst_label.setFont(dst_font)
        self._dst_label.setStyleSheet("color: #66FFB2; margin: 0; padding: 0;")

        # 透明度效果元件
        self._src_effect = QGraphicsOpacityEffect(self._src_label)
        self._src_effect.setOpacity(1.0)
        self._src_label.setGraphicsEffect(self._src_effect)
        self._dst_effect = QGraphicsOpacityEffect(self._dst_label)
        self._dst_effect.setOpacity(1.0)
        self._dst_label.setGraphicsEffect(self._dst_effect)

    # ---- Tray icon --------------------------------------------------------

    def _init_tray(self):
        self._tray = QSystemTrayIcon(self)
        # Use a simple built-in icon or fallback
        icon = QIcon.fromTheme("media-captions")
        if icon.isNull():
            icon = self.style().standardIcon(
                self.style().StandardPixmap.SP_ComputerIcon
            )
        self._tray.setIcon(icon)
        self._tray.setToolTip("jt-live-whisper 字幕浮層")

        menu = QMenu()

        # Click-through toggle
        self._ct_action = QAction("滑鼠穿透模式", self)
        self._ct_action.setCheckable(True)
        self._ct_action.setChecked(self._click_through)
        self._ct_action.triggered.connect(self._toggle_click_through)
        menu.addAction(self._ct_action)
        menu.addSeparator()

        # Font size
        font_menu = menu.addMenu("字體大小")
        for key, label in [("small", "小"), ("medium", "中"), ("large", "大")]:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(key == self._font_preset)
            act.triggered.connect(lambda checked, k=key: self._set_font_preset(k))
            font_menu.addAction(act)
        self._font_actions = font_menu.actions()

        # Opacity
        opacity_menu = menu.addMenu("背景透明度")
        for pct in (50, 65, 80):
            act = QAction(f"{pct}%", self)
            act.setCheckable(True)
            act.setChecked(pct == self._cfg.get("opacity", 65))
            act.triggered.connect(lambda checked, p=pct: self._set_opacity(p))
            opacity_menu.addAction(act)
        self._opacity_actions = opacity_menu.actions()

        menu.addSeparator()
        quit_act = QAction("結束字幕浮層", self)
        quit_act.triggered.connect(self.close)
        menu.addAction(quit_act)

        self._tray.setContextMenu(menu)
        self._tray.show()

    # ---- Timers -----------------------------------------------------------

    def _init_timers(self):
        # Reconnect timer
        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setInterval(RECONNECT_MS)
        self._reconnect_timer.timeout.connect(self._connect)

        # Idle fade timer
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.setInterval(IDLE_FADE_SECS * 1000)
        self._idle_timer.timeout.connect(self._on_idle)

    # ---- Connection -------------------------------------------------------

    def _connect(self):
        """嘗試 WebSocket 連線，失敗則回退 TCP"""
        self._reconnect_timer.stop()

        if HAS_WEBSOCKET:
            self._connect_ws()
        else:
            self._connect_tcp()

    def _connect_ws(self):
        if self._ws is not None:
            self._ws.close()
            self._ws.deleteLater()
        self._ws = QWebSocket()
        self._ws.textMessageReceived.connect(self._on_ws_message)
        self._ws.connected.connect(self._on_ws_connected)
        self._ws.disconnected.connect(self._on_disconnected)
        self._ws.open(QUrl(self._ws_url))

    def _connect_tcp(self):
        if self._tcp is not None:
            self._tcp.close()
            self._tcp.deleteLater()
        self._tcp = QTcpSocket(self)
        self._tcp.readyRead.connect(self._on_tcp_data)
        self._tcp.connected.connect(self._on_tcp_connected)
        self._tcp.disconnected.connect(self._on_disconnected)
        self._tcp.errorOccurred.connect(self._on_tcp_error)
        self._tcp_buf = b""
        self._tcp.connectToHost(self._tcp_host, self._tcp_port)

    def _on_ws_connected(self):
        self._connected = True
        self._reconnect_timer.stop()
        self._dst_label.setText("")
        self._src_label.setText("")

    def _on_tcp_connected(self):
        self._connected = True
        self._reconnect_timer.stop()
        self._dst_label.setText("")
        self._src_label.setText("")

    def _on_disconnected(self):
        self._connected = False
        self._dst_label.setText("連線中...")
        self._src_label.setText("")
        self._reconnect_timer.start()

    def _on_tcp_error(self, error):
        if not self._connected:
            # WS fallback: if WS failed, try TCP; if TCP also failed, schedule reconnect
            if self._ws is not None:
                # First attempt was WS; now try TCP
                self._ws.close()
                self._ws.deleteLater()
                self._ws = None
                self._connect_tcp()
                return
            self._reconnect_timer.start()

    # ---- Message handling -------------------------------------------------

    def _on_ws_message(self, text: str):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        self._handle_event(data)

    def _on_tcp_data(self):
        self._tcp_buf += bytes(self._tcp.readAll())
        while b"\n" in self._tcp_buf:
            line, self._tcp_buf = self._tcp_buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            self._handle_event(data)

    def _handle_event(self, data: dict):
        if data.get("type") == "keyword_alert" and data.get("overlay_flash"):
            self._flash_border()
            return
        if data.get("type") != "transcription":
            return

        src_text = data.get("src_text", "")
        dst_text = data.get("dst_text", "")

        if self._single_line:
            new_src = ""
            new_dst = dst_text or src_text
        elif src_text and dst_text:
            new_src = src_text
            new_dst = dst_text
        elif src_text:
            new_src = ""
            new_dst = src_text
        else:
            return

        # 動態偵測單語/雙語：根據實際有無 dst_text 調整高度
        has_both = bool(src_text and dst_text)
        if has_both != self._is_bilingual:
            self._is_bilingual = has_both
            geo = self.geometry()
            new_h = 80 if has_both else 42
            geo.setTop(geo.bottom() - new_h)
            self.setGeometry(geo)

        # 淡出 → 更新文字 → 淡入
        self._pending_src = new_src
        self._pending_dst = new_dst
        self._crossfade_out()

        # Reset idle state → full opacity
        self._idle = False
        self._target_bg_alpha = self._bg_alpha
        self.update()

        # Restart idle timer
        self._idle_timer.start()

    def _crossfade_out(self):
        """淡出文字（120ms）"""
        # 停止之前的動畫避免堆疊
        for a in ('_fade_out_anim_src','_fade_out_anim_dst','_fade_in_anim_src','_fade_in_anim_dst'):
            old = getattr(self, a, None)
            if old: old.stop()
        self._fade_out_anim_src = QPropertyAnimation(self._src_effect, b"opacity")
        self._fade_out_anim_src.setDuration(120)
        self._fade_out_anim_src.setStartValue(self._src_effect.opacity())
        self._fade_out_anim_src.setEndValue(0.0)

        self._fade_out_anim_dst = QPropertyAnimation(self._dst_effect, b"opacity")
        self._fade_out_anim_dst.setDuration(120)
        self._fade_out_anim_dst.setStartValue(self._dst_effect.opacity())
        self._fade_out_anim_dst.setEndValue(0.0)
        self._fade_out_anim_dst.finished.connect(self._crossfade_update)

        self._fade_out_anim_src.start()
        self._fade_out_anim_dst.start()

    def _crossfade_update(self):
        """淡出完成 → 更新文字 → 淡入"""
        self._src_label.setText(self._pending_src)
        self._dst_label.setText(self._pending_dst)
        # 單語時隱藏原文 label（不佔空間），雙語時顯示
        self._src_label.setVisible(bool(self._pending_src))
        self._adjust_height()

        # 淡入（180ms）
        self._fade_in_anim_src = QPropertyAnimation(self._src_effect, b"opacity")
        self._fade_in_anim_src.setDuration(180)
        self._fade_in_anim_src.setStartValue(0.0)
        self._fade_in_anim_src.setEndValue(1.0)
        self._fade_in_anim_src.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._fade_in_anim_dst = QPropertyAnimation(self._dst_effect, b"opacity")
        self._fade_in_anim_dst.setDuration(180)
        self._fade_in_anim_dst.setStartValue(0.0)
        self._fade_in_anim_dst.setEndValue(1.0)
        self._fade_in_anim_dst.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._fade_in_anim_src.start()
        self._fade_in_anim_dst.start()

    def _adjust_height(self):
        """依視窗大小自動計算字體，最小不低於 MIN_FONT，容不下則換行"""
        MIN_SRC_FONT = 10
        MIN_DST_FONT = 12
        avail_h = self.height() - 8  # contentsMargins (4+4)
        avail_w = self.width() - 40

        # 基準字體按視窗高度比例計算（視窗越大字越大）
        dst_base = max(MIN_DST_FONT, int(self.height() * 0.22))
        src_base = max(MIN_SRC_FONT, int(dst_base * 0.7))

        font_family = self._cfg.get("font_family", "")

        # 逐步縮小直到放得下（最小到 MIN）
        for shrink in range(0, dst_base - MIN_DST_FONT + 1):
            src_size = max(MIN_SRC_FONT, src_base - shrink)
            dst_size = max(MIN_DST_FONT, dst_base - shrink)
            _mk_font = lambda sz: QFont(font_family, sz) if font_family else QFont(QFont().family(), sz)
            src_fm = QFontMetrics(_mk_font(src_size))
            dst_fm = QFontMetrics(_mk_font(dst_size))
            src_h = src_fm.boundingRect(0, 0, avail_w, 0,
                        Qt.TextFlag.TextWordWrap, self._src_label.text() or "X").height() if self._src_label.text() else 0
            dst_h = dst_fm.boundingRect(0, 0, avail_w, 0,
                        Qt.TextFlag.TextWordWrap, self._dst_label.text() or "X").height()
            if src_h + dst_h <= avail_h:
                break

        # 套用字體（已到最小還放不下就靠 WordWrap 換行）
        sf = self._src_label.font()
        sf.setPointSize(src_size)
        self._src_label.setFont(sf)
        df = self._dst_label.font()
        df.setPointSize(dst_size)
        self._dst_label.setFont(df)

    def _flash_border(self):
        """關鍵字匹配時邊框閃爍"""
        # 停止之前的閃爍 timer
        old = getattr(self, '_flash_timer2', None)
        if old: old.stop()
        self._flash_count = 0
        self._flash_on = True
        self._border_color = QColor(255, 193, 7)  # 金黃色
        self._flash_timer2 = QTimer(self)
        self._flash_timer2.setInterval(200)
        self._flash_timer2.timeout.connect(self._flash_step)
        self._flash_timer2.start()
        self.update()

    def _flash_step(self):
        self._flash_count += 1
        self._flash_on = not self._flash_on
        self.update()
        if self._flash_count >= 6:  # 3 次閃爍
            self._flash_timer2.stop()
            self._border_color = None
            self.update()

    def _on_idle(self):
        """閒置後降低透明度"""
        self._idle = True
        self._target_bg_alpha = IDLE_OPACITY
        self.update()

    # ---- Painting ---------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        alpha = self._target_bg_alpha * self._bg_opacity
        color = QColor(20, 20, 25, int(255 * alpha))
        painter.setBrush(QBrush(color))

        # 關鍵字閃爍邊框
        border_color = getattr(self, '_border_color', None)
        if border_color and getattr(self, '_flash_on', False):
            painter.setPen(QPen(border_color, 3))
        else:
            painter.setPen(Qt.PenStyle.NoPen)

        rect = QRectF(0, 0, self.width(), self.height())
        painter.drawRoundedRect(rect, 12, 12)
        painter.end()

    # ---- Mouse interaction ------------------------------------------------

    _EDGE_MARGIN = 8  # 邊緣拖拉區域寬度

    def _detect_edge(self, pos):
        """偵測滑鼠位置是否在邊緣（回傳 'left','right','top','bottom','topleft' 等，或 None）"""
        m = self._EDGE_MARGIN
        r = self.rect()
        edges = []
        if pos.y() <= m: edges.append('top')
        if pos.y() >= r.height() - m: edges.append('bottom')
        if pos.x() <= m: edges.append('left')
        if pos.x() >= r.width() - m: edges.append('right')
        return ''.join(edges) if edges else None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            edge = self._detect_edge(event.position().toPoint())
            if edge:
                self._resize_edge = edge
                self._resize_origin = event.globalPosition().toPoint()
                self._resize_geo = self.geometry()
            else:
                self._resize_edge = None
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            # 只更新游標形狀
            edge = self._detect_edge(event.position().toPoint())
            if edge in ('left', 'right'):
                self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
            elif edge in ('top', 'bottom'):
                self.setCursor(QCursor(Qt.CursorShape.SizeVerCursor))
            elif edge in ('topleft', 'bottomright'):
                self.setCursor(QCursor(Qt.CursorShape.SizeFDiagCursor))
            elif edge in ('topright', 'bottomleft'):
                self.setCursor(QCursor(Qt.CursorShape.SizeBDiagCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            return

        if self._resize_edge:
            delta = event.globalPosition().toPoint() - self._resize_origin
            geo = QRectF(self._resize_geo)
            if 'right' in self._resize_edge:
                geo.setRight(geo.right() + delta.x())
            if 'left' in self._resize_edge:
                geo.setLeft(geo.left() + delta.x())
            if 'bottom' in self._resize_edge:
                geo.setBottom(geo.bottom() + delta.y())
            if 'top' in self._resize_edge:
                geo.setTop(geo.top() + delta.y())
            if geo.width() >= self.minimumWidth() and geo.height() >= self.minimumHeight():
                self.setGeometry(geo.toRect())
            event.accept()
        elif self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        self._resize_edge = None
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # X 按鈕跟隨右上角
        self._close_btn.move(self.width() - 30, 6)
        # 字體自動縮放
        if hasattr(self, '_src_label') and (self._src_label.text() or self._dst_label.text()):
            self._adjust_height()

    def enterEvent(self, event):
        self._hover = True
        self._close_btn.move(self.width() - 30, 6)
        if not self._click_through:
            self._close_btn.show()

    def leaveEvent(self, event):
        self._hover = False
        self._close_btn.hide()
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    # ---- Click-through ----------------------------------------------------

    def _toggle_click_through(self, checked):
        self._click_through = checked
        self._apply_click_through()

    def _apply_click_through(self):
        if self._click_through:
            if IS_WINDOWS:
                self._set_win_click_through(True)
            else:
                self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self._close_btn.hide()
        else:
            if IS_WINDOWS:
                self._set_win_click_through(False)
            else:
                self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    def _set_win_click_through(self, enable: bool):
        """Windows: 透過 ctypes 設定 WS_EX_TRANSPARENT"""
        if not IS_WINDOWS:
            return
        try:
            import ctypes
            hwnd = int(self.winId())
            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_LAYERED = 0x00080000
            user32 = ctypes.windll.user32
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            if enable:
                style |= WS_EX_TRANSPARENT | WS_EX_LAYERED
            else:
                style &= ~WS_EX_TRANSPARENT
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception:
            pass

    # ---- Font / opacity menu actions --------------------------------------

    def _set_font_preset(self, key: str):
        self._font_preset = key
        for act in self._font_actions:
            act.setChecked(False)
        for act in self._font_actions:
            label_map = {"小": "small", "中": "medium", "大": "large"}
            if label_map.get(act.text()) == key:
                act.setChecked(True)
        self._apply_fonts()

    def _set_opacity(self, pct: int):
        self._bg_alpha = pct / 100.0
        self._target_bg_alpha = self._bg_alpha
        self._cfg["opacity"] = pct
        for act in self._opacity_actions:
            act.setChecked(act.text() == f"{pct}%")
        self.update()

    # ---- Position save / restore ------------------------------------------

    def _config_path(self) -> str:
        return self._cfg.get("config_path", os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.json"
        ))

    def _save_position(self):
        try:
            path = self._config_path()
            cfg = {}
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            section = cfg.setdefault(CONFIG_SECTION, {})
            geo = self.geometry()
            section["x"] = geo.x()
            section["y"] = geo.y()
            section["font_preset"] = self._font_preset
            section["opacity"] = self._cfg.get("opacity", 65)
            section["click_through"] = self._click_through
            section["single_line"] = self._single_line
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _restore_position(self):
        try:
            path = self._config_path()
            if not os.path.isfile(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            section = cfg.get(CONFIG_SECTION, {})
            if "x" in section and "y" in section:
                self.move(section["x"], section["y"])
        except Exception:
            pass

    # ---- Close ------------------------------------------------------------

    def closeEvent(self, event):
        self._save_position()
        if self._ws is not None:
            self._ws.close()
        if self._tcp is not None:
            self._tcp.close()
        self._reconnect_timer.stop()
        self._idle_timer.stop()
        self._tray.hide()
        event.accept()
        QApplication.quit()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | None) -> dict:
    """從 config.json 讀取 subtitle_overlay 設定"""
    cfg: dict = {}
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.json"
        )
    cfg["config_path"] = config_path

    try:
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            section = data.get(CONFIG_SECTION, {})
            cfg.update(section)
    except Exception:
        pass

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser(
        description="jt-live-whisper 桌面字幕浮層"
    )
    parser.add_argument("--ws-url", default=None,
                        help=f"WebSocket 網址（預設 {DEFAULT_WS_URL}）")
    parser.add_argument("--tcp-host", default=None,
                        help=f"TCP 主機（預設 {DEFAULT_TCP_HOST}）")
    parser.add_argument("--tcp-port", type=int, default=None,
                        help=f"TCP 埠號（預設 {DEFAULT_TCP_PORT}）")
    parser.add_argument("--config", default=None,
                        help="config.json 路徑")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.ws_url:
        cfg["ws_url"] = args.ws_url
    if args.tcp_host:
        cfg["tcp_host"] = args.tcp_host
    if args.tcp_port:
        cfg["tcp_port"] = args.tcp_port

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    overlay = SubtitleOverlay(cfg)
    overlay.show()
    overlay.raise_()

    # macOS: 用原生 API 設定視窗層級，確保永遠在所有視窗之上
    if IS_MACOS:
        try:
            import ctypes
            import ctypes.util
            objc_lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
            objc_lib.objc_getClass.restype = ctypes.c_void_p
            objc_lib.sel_registerName.restype = ctypes.c_void_p
            objc_lib.objc_msgSend.restype = ctypes.c_void_p
            objc_lib.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

            # 取得 NSView → NSWindow
            nsview_ptr = int(overlay.winId())
            sel_window = objc_lib.sel_registerName(b"window")
            nswindow = objc_lib.objc_msgSend(nsview_ptr, sel_window)
            if nswindow:
                # setLevel: 25 = NSStatusWindowLevel（比 NSFloatingWindowLevel 更高）
                sel_setLevel = objc_lib.sel_registerName(b"setLevel:")
                objc_setLevel = objc_lib.objc_msgSend
                objc_setLevel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
                objc_setLevel(nswindow, sel_setLevel, 25)

                # setCollectionBehavior: canJoinAllSpaces | fullScreenAuxiliary | stationary
                sel_setBehavior = objc_lib.sel_registerName(b"setCollectionBehavior:")
                objc_setBehavior = objc_lib.objc_msgSend
                objc_setBehavior.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
                behavior = (1 << 0) | (1 << 8) | (1 << 4)  # canJoinAllSpaces | fullScreenAuxiliary | stationary
                objc_setBehavior(nswindow, sel_setBehavior, behavior)
        except Exception as e:
            print(f"[懸浮字幕] macOS 視窗層級設定失敗: {e}", file=sys.stderr)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
