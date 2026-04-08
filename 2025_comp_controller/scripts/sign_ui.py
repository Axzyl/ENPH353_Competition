#!/usr/bin/env python3
"""
sign_ui.py
----------
Standalone UI that displays sign reading results pushed from main.py.

main.py calls sign_ui.push(frame, sign_roi, top_text, bot_text)
to update the display each time a sign is read.

The UI shows:
  - Full camera frame (with detection box)
  - Extracted sign ROI
  - Top half of sign
  - Bottom half of sign
  - OCR text for each half
  - History of all readings in a scrollable log

Usage:
    # In main.py (import and call push):
    import sign_ui
    sign_ui.init()   # call once after rospy.init_node
    ...
    sign_ui.push(frame, sign_roi, top_text, bot_text)

    # Or run standalone to see a demo:
    python3 sign_ui.py
"""

import base64
import threading
import time
import tkinter as tk
from tkinter import scrolledtext
import cv2
import numpy as np


# ------------------------------------------------------------------ #
# Layout constants                                                     #
# ------------------------------------------------------------------ #

CAM_W,  CAM_H  = 480, 270    # camera frame thumbnail
SIGN_W, SIGN_H = 280, 160    # extracted sign ROI
HALF_W, HALF_H = 136, 76     # top/bottom half thumbnails

BG        = "#0e0e0e"
SURFACE   = "#161616"
CARD      = "#1c1c1c"
BORDER    = "#2a2a2a"
ACCENT    = "#c8f542"         # sharp lime green
ACCENT2   = "#f5a623"         # amber for bottom
TEXT      = "#e8e8e8"
MUTED     = "#555555"
FONT_MONO = ("Courier", 9)
FONT_HEAD = ("Courier", 11, "bold")
FONT_BIG  = ("Courier", 20, "bold")

# ------------------------------------------------------------------ #
# State                                                                #
# ------------------------------------------------------------------ #

_root        = None
_inited      = False
_tk_thread   = None
_history     = []      # list of (top_text, bot_text, timestamp)
_push_lock   = threading.Lock()

# Widget refs
_cam_canvas  = None
_sign_canvas = None
_top_canvas  = None
_bot_canvas  = None
_lbl_top     = None
_lbl_bot     = None
_lbl_count   = None
_log_box     = None


# ------------------------------------------------------------------ #
# Image helpers                                                        #
# ------------------------------------------------------------------ #

def _to_tk(bgr, w, h):
    """Resize and encode a BGR image as a tk.PhotoImage."""
    if bgr is None or bgr.size == 0:
        return None
    resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    _, buf  = cv2.imencode(".png", resized)
    b64     = base64.b64encode(buf.tobytes())
    return tk.PhotoImage(data=b64)


def _show(canvas, tkimg, w, h):
    """Put a tk.PhotoImage centred on a canvas."""
    canvas.delete("all")
    if tkimg is None:
        canvas.create_text(w // 2, h // 2,
                           text="—", fill=MUTED,
                           font=("Courier", 14))
        return
    canvas.create_image(0, 0, anchor="nw", image=tkimg)
    canvas._img_ref = tkimg   # prevent GC


def _placeholder(canvas, w, h, label=""):
    canvas.delete("all")
    canvas.create_rectangle(0, 0, w, h, fill=CARD, outline=BORDER)
    canvas.create_text(w // 2, h // 2,
                       text=label or "—", fill=MUTED,
                       font=("Courier", 10))


# ------------------------------------------------------------------ #
# UI build                                                             #
# ------------------------------------------------------------------ #

def _build_ui():
    global _root
    global _cam_canvas, _sign_canvas, _top_canvas, _bot_canvas
    global _lbl_top, _lbl_bot, _lbl_count, _log_box

    _root = tk.Tk()
    _root.title("Sign Reader")
    _root.configure(bg=BG)
    _root.resizable(False, False)

    # ── header ──────────────────────────────────────────────────────
    hdr = tk.Frame(_root, bg=BG)
    hdr.pack(fill=tk.X, padx=16, pady=(14, 6))

    tk.Label(hdr, text="SIGN READER",
             bg=BG, fg=ACCENT,
             font=("Courier", 15, "bold")).pack(side=tk.LEFT)

    _lbl_count = tk.Label(hdr, text="0 readings",
                          bg=BG, fg=MUTED,
                          font=FONT_MONO)
    _lbl_count.pack(side=tk.RIGHT)

    # ── main row ────────────────────────────────────────────────────
    row = tk.Frame(_root, bg=BG)
    row.pack(padx=16, pady=4)

    # LEFT: camera frame
    left = tk.Frame(row, bg=BG)
    left.pack(side=tk.LEFT, padx=(0, 12))

    _section_label(left, "CAMERA")
    _cam_canvas = tk.Canvas(left, width=CAM_W, height=CAM_H,
                            bg=CARD, highlightthickness=1,
                            highlightbackground=BORDER)
    _cam_canvas.pack()
    _placeholder(_cam_canvas, CAM_W, CAM_H, "awaiting capture")

    # RIGHT: sign panels
    right = tk.Frame(row, bg=BG)
    right.pack(side=tk.LEFT)

    # Sign ROI
    _section_label(right, "EXTRACTED SIGN")
    _sign_canvas = tk.Canvas(right, width=SIGN_W, height=SIGN_H,
                             bg=CARD, highlightthickness=1,
                             highlightbackground=BORDER)
    _sign_canvas.pack()
    _placeholder(_sign_canvas, SIGN_W, SIGN_H)

    # Halves row
    halves_row = tk.Frame(right, bg=BG)
    halves_row.pack(pady=(8, 0))

    # Top half
    top_col = tk.Frame(halves_row, bg=BG)
    top_col.pack(side=tk.LEFT, padx=(0, 8))
    _section_label(top_col, "TOP HALF")
    _top_canvas = tk.Canvas(top_col, width=HALF_W, height=HALF_H,
                            bg=CARD, highlightthickness=1,
                            highlightbackground=BORDER)
    _top_canvas.pack()
    _placeholder(_top_canvas, HALF_W, HALF_H)

    # Top OCR
    _lbl_top = tk.Label(top_col, text="",
                        bg=BG, fg=ACCENT,
                        font=("Courier", 12, "bold"),
                        wraplength=HALF_W, justify="center")
    _lbl_top.pack(pady=(4, 0))

    # Bottom half
    bot_col = tk.Frame(halves_row, bg=BG)
    bot_col.pack(side=tk.LEFT)
    _section_label(bot_col, "BOTTOM HALF")
    _bot_canvas = tk.Canvas(bot_col, width=HALF_W, height=HALF_H,
                            bg=CARD, highlightthickness=1,
                            highlightbackground=BORDER)
    _bot_canvas.pack()
    _placeholder(_bot_canvas, HALF_W, HALF_H)

    # Bottom OCR
    _lbl_bot = tk.Label(bot_col, text="",
                        bg=BG, fg=ACCENT2,
                        font=("Courier", 12, "bold"),
                        wraplength=HALF_W, justify="center")
    _lbl_bot.pack(pady=(4, 0))

    # ── log ─────────────────────────────────────────────────────────
    log_frame = tk.Frame(_root, bg=BG)
    log_frame.pack(fill=tk.X, padx=16, pady=(10, 14))

    _section_label(log_frame, "READING LOG")
    _log_box = scrolledtext.ScrolledText(
        log_frame, height=6,
        bg=SURFACE, fg=MUTED,
        font=FONT_MONO,
        state=tk.DISABLED,
        relief=tk.FLAT, bd=0,
        highlightthickness=1,
        highlightbackground=BORDER
    )
    _log_box.pack(fill=tk.X)

    _root.protocol("WM_DELETE_WINDOW", lambda: None)   # keep open
    _root.mainloop()


def _section_label(parent, text):
    tk.Label(parent, text=text,
             bg=BG, fg=MUTED,
             font=("Courier", 8, "bold")).pack(anchor="w", pady=(0, 3))


# ------------------------------------------------------------------ #
# Update helpers (all called via root.after from any thread)           #
# ------------------------------------------------------------------ #

def _do_update(frame, sign_roi, top_crop, bot_crop, top_text, bot_text, timestamp):
    """Execute all UI updates on the tkinter thread."""
    global _history

    # Camera frame
    tk_cam = _to_tk(frame, CAM_W, CAM_H)
    _show(_cam_canvas, tk_cam, CAM_W, CAM_H)

    # Sign ROI
    if sign_roi is not None and sign_roi.size > 0:
        tk_sign = _to_tk(sign_roi, SIGN_W, SIGN_H)
        _show(_sign_canvas, tk_sign, SIGN_W, SIGN_H)

        # Show cropped halves directly from sign_reader
        if top_crop is not None and top_crop.size > 0:
            _show(_top_canvas, _to_tk(top_crop, HALF_W, HALF_H), HALF_W, HALF_H)
        else:
            _placeholder(_top_canvas, HALF_W, HALF_H)

        if bot_crop is not None and bot_crop.size > 0:
            _show(_bot_canvas, _to_tk(bot_crop, HALF_W, HALF_H), HALF_W, HALF_H)
        else:
            _placeholder(_bot_canvas, HALF_W, HALF_H)
    else:
        _placeholder(_sign_canvas, SIGN_W, SIGN_H, "no sign")
        _placeholder(_top_canvas, HALF_W, HALF_H)
        _placeholder(_bot_canvas, HALF_W, HALF_H)

    # Text labels
    _lbl_top.config(text=top_text or "—")
    _lbl_bot.config(text=bot_text or "—")

    # Count
    _history.append((top_text, bot_text, timestamp))
    _lbl_count.config(text=f"{len(_history)} reading{'s' if len(_history) != 1 else ''}")

    # Log entry
    _log_box.config(state=tk.NORMAL)
    _log_box.insert(tk.END,
                    f"[{timestamp}]  TOP: {top_text!r:20}  BOT: {bot_text!r}\n")
    _log_box.see(tk.END)
    _log_box.config(state=tk.DISABLED)


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def init():
    """
    Launch the UI window in a background thread.
    Call once — safe to call multiple times (idempotent).
    """
    global _inited, _tk_thread
    if _inited:
        return
    _tk_thread = threading.Thread(target=_build_ui, daemon=True)
    _tk_thread.start()
    time.sleep(0.6)   # let window initialise
    _inited = True


def push(frame, sign_roi, top_crop, bot_crop, top_text, bot_text):
    """
    Push a new sign reading to the UI.

    Args:
        frame     : BGR numpy array — full camera frame
        sign_roi  : BGR numpy array — extracted sign, or None
        top_text  : OCR result from top half
        bot_text  : OCR result from bottom half
    """
    if _root is None:
        return
    ts = time.strftime("%H:%M:%S")
    f  = frame.copy()     if frame     is not None else None
    s  = sign_roi.copy()  if sign_roi  is not None else None
    tc = top_crop.copy()  if top_crop  is not None else None
    bc = bot_crop.copy()  if bot_crop  is not None else None
    with _push_lock:
        _root.after(0, lambda: _do_update(f, s, tc, bc, top_text, bot_text, ts))


# ------------------------------------------------------------------ #
# Integrate with main.py read_sign                                     #
# ------------------------------------------------------------------ #

def wrap_read_sign(read_sign_fn, camera="center", **kwargs):
    """
    Convenience wrapper — calls read_sign and pushes result to UI.

    Usage in main.py:
        import sign_ui
        sign_ui.init()
        top, bot = sign_ui.wrap_read_sign(read_sign, "left")
    """
    import sign_reader as _sr

    # Temporarily hook into sign_reader to capture the frame + roi
    _captured = {}

    _orig_cb = _sr._result_callback

    def _hook(roi, t, b):
        _captured["roi"] = roi
        if _orig_cb:
            _orig_cb(roi, t, b)

    _sr.set_result_callback(_hook)
    top, bot = read_sign_fn(camera, **kwargs)
    _sr.set_result_callback(_orig_cb)

    frame = _captured.get("roi")   # use roi as frame proxy if full frame unavailable
    push(frame, _captured.get("roi"), top, bot)
    return top, bot


# ------------------------------------------------------------------ #
# Standalone demo                                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    init()
    time.sleep(1.0)

    print("Pushing demo readings...")

    # Demo frame — solid grey
    demo_frame = np.full((480, 640, 3), 60, dtype=np.uint8)
    cv2.putText(demo_frame, "DEMO FRAME", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (180, 180, 180), 2)

    # Demo sign
    demo_sign = np.full((120, 280, 3), 230, dtype=np.uint8)
    cv2.putText(demo_sign, "SIZE",      (80,  45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 200), 2)
    cv2.putText(demo_sign, "FIVE",      (70,  95),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 200), 2)

    push(demo_frame, demo_sign, "SIZE", "FIVE")
    time.sleep(2.0)
    push(demo_frame, demo_sign, "TIME", "FINALS WEEK")

    print("Demo pushed. Close the window to exit.")
    input("Press Enter to quit...")