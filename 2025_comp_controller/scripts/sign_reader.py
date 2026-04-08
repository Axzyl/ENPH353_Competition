#!/usr/bin/env python3
"""
sign_reader.py
--------------
Detects blue-bordered signs in a camera frame, extracts the white
sign interior, splits it vertically, and runs Tesseract OCR on each half.

No neural networks used. Requires:
    sudo apt install tesseract-ocr
    pip3 install pytesseract --user

Public API:
    sign_reader.process_frame(bgr_frame)
        -> (annotated_frame, sign_roi, top_crop, bot_crop, top_text, bot_text)
        -> (annotated_frame, None, None, None, "", "")  if no sign found

    sign_reader.get_last_result()
        -> (sign_roi, top_crop, bot_crop, top_text, bot_text)
"""

import cv2
import numpy as np
import pytesseract

# ------------------------------------------------------------------ #
# Tuning constants                                                     #
# ------------------------------------------------------------------ #

SIGN_BLUE_LO = np.array([100, 120, 80])
SIGN_BLUE_HI = np.array([130, 255, 220])

MIN_SIGN_AREA = 2000
BLUE_FRAC_MIN = 0.05
BLUE_FRAC_MAX = 0.55
ASPECT_MIN    = 0.5
ASPECT_MAX    = 4.0
OCR_SCALE     = 1

TESS_CONFIG = ("--oem 0 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")

ANNO_COLOR = (0, 255, 0)

# ------------------------------------------------------------------ #
# Module-level result cache                                            #
# ------------------------------------------------------------------ #

_last_sign_roi  = None
_last_top_crop  = None
_last_bot_crop  = None
_last_top_text  = ""
_last_bot_text  = ""

_result_callback = None


def set_result_callback(fn):
    """fn(sign_roi, top_crop, bot_crop, top_text, bot_text)"""
    global _result_callback
    _result_callback = fn


def get_last_result():
    return _last_sign_roi, _last_top_crop, _last_bot_crop, _last_top_text, _last_bot_text


# ================================================================== #
# SIGN DETECTION                                                      #
# ================================================================== #

def _blue_mask(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, SIGN_BLUE_LO, SIGN_BLUE_HI)
    k    = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _find_best_sign(frame, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best_area = 0
    best_rect = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < MIN_SIGN_AREA:
            continue
        aspect = w / max(h, 1)
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue
        roi_mask  = mask[y:y + h, x:x + w]
        blue_frac = np.sum(roi_mask > 0) / area
        if not (BLUE_FRAC_MIN <= blue_frac <= BLUE_FRAC_MAX):
            continue
        if area > best_area:
            best_area = area
            best_rect = (x, y, w, h)

    return best_rect


def _extract_interior(frame, rect):
    x, y, w, h = rect
    border = max(4, int(min(w, h) * 0.08))
    ix = x + border
    iy = y + border
    iw = w - 2 * border
    ih = h - 2 * border
    if iw < 10 or ih < 10:
        return frame[y:y + h, x:x + w]
    return frame[iy:iy + ih, ix:ix + iw]


# ================================================================== #
# CROPPING                                                            #
# ================================================================== #

TRANSITION_THRESH = 20   # min brightness change to count as a border transition
TRANSITION_WINDOW = 8    # compare pixels this many steps apart to handle gradients
VERTICAL_SCAN_OFFSET = 10  # pixels inward from left/right border for vertical scan

def _first_transition(values, forward=True,
                      thresh=TRANSITION_THRESH, window=TRANSITION_WINDOW):
    """
    Find the FIRST position from the edge where the brightness changes
    by at least thresh over a window of `window` pixels.

    Using a window (instead of adjacent pixel diffs) handles gradient
    borders where no single step exceeds the threshold.

    Scans from left if forward=True, from right if forward=False.
    Returns index just inside the transition in original coordinates.
    """
    seq = values if forward else values[::-1]
    arr = seq.astype(np.int32)
    n   = len(arr)
    for i in range(n - window):
        if abs(int(arr[i + window]) - int(arr[i])) >= thresh:
            # Transition starts at i, content begins at i + window
            result = i + window
            return result if forward else n - result - 1
    return 0 if forward else n - 1


def _crop_to_white_border(img, min_size=10):
    """
    Crop the blue border by detecting colour transitions rather than
    absolute thresholds.

    1. At mid_y, scan horizontally — find the largest brightness jump
       from the left edge (left boundary) and from the right edge
       (right boundary).
    2. At left_x, scan vertically — find the largest jump going up
       (top-left corner) and going down (bottom-left corner).
    3. At right_x, scan vertically — find top-right and bottom-right.
    4. top  = max(top_left,  top_right)   — conservative inner bound
       bot  = min(bot_left,  bot_right)   — conservative inner bound
    5. Crop to (left, top, right, bot).
    """
    if len(img.shape) != 3:
        return img

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape
    mid_y = h // 2

    # ---- Step 1: Horizontal scan at mid_y ---- #
    row = gray[mid_y, :]
    left  = _first_transition(row, forward=True)
    right = _first_transition(row, forward=False)

    if right <= left or (right - left) < min_size:
        return img

    # ---- Step 2 & 3: Vertical scans ---- #
    # Move one pixel inward from each border to start on white
    col_left  = gray[:, min(left + 1, right)]
    col_right = gray[:, max(right - 1, left)]

    # Top: scan downward from top edge — first transition = top border end
    top_left  = _first_transition(col_left[:mid_y + 1],  forward=True)
    top_right = _first_transition(col_right[:mid_y + 1], forward=True)

    # Bottom: scan upward from bottom edge — first transition = bottom border end
    bot_left  = mid_y + _first_transition(col_left[mid_y:],  forward=False)
    bot_right = mid_y + _first_transition(col_right[mid_y:], forward=False)

    # ---- Step 4: Conservative inner bounding box ---- #
    top = max(top_left, top_right)
    bot = min(bot_left, bot_right)

    if (bot - top) < min_size or (right - left) < min_size:
        return img

    return img[top:bot, left:right + 1]


# ================================================================== #
# OCR                                                                 #
# ================================================================== #

def _preprocess_for_ocr(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (w * OCR_SCALE, h * OCR_SCALE),
                      interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 128:
        binary = cv2.bitwise_not(binary)
    return binary


def _run_ocr(binary_img):
    try:
        raw     = pytesseract.image_to_string(binary_img, config=TESS_CONFIG)
        cleaned = " ".join(
            "".join(c for c in raw if c.isalpha() or c.isdigit() or c == " ").split()
        )
        return cleaned
    except Exception as e:
        return f"[OCR error: {e}]"


def _read_sign(sign_roi):
    """
    Split sign ROI into halves, crop each to its white border,
    OCR each half. Returns (top_crop, bot_crop, top_text, bot_text).
    """
    h, w = sign_roi.shape[:2]
    mid  = h // 2

    top_half = sign_roi[:mid, :]
    bot_half = sign_roi[mid:, :]

    # Crop each half tightly to its blue border
    top_crop = _crop_to_white_border(top_half)
    bot_crop = _crop_to_white_border(bot_half)

    # Debug: save images to /tmp to verify cropping
    cv2.imwrite("/tmp/sign_top_half_raw.png", top_half)
    cv2.imwrite("/tmp/sign_bot_half_raw.png", bot_half)
    cv2.imwrite("/tmp/sign_top_crop.png", top_crop)
    cv2.imwrite("/tmp/sign_bot_crop.png", bot_crop)

    top_bin  = _preprocess_for_ocr(top_crop)
    bot_bin  = _preprocess_for_ocr(bot_crop)

    # top_text = _run_ocr(top_bin)
    top_text = "temp"
    bot_text = _run_ocr(bot_bin)

    return top_crop, bot_crop, top_text, bot_text


# ================================================================== #
# MAIN PROCESSOR                                                      #
# ================================================================== #

def process_frame(frame):
    """
    Detect the blue-bordered sign in `frame`, extract its interior,
    crop halves to white border, run OCR on each half.

    Returns:
        annotated_frame : frame with detection box drawn
        sign_roi        : cropped BGR sign interior, or None
        top_crop        : top half cropped to white border, or None
        bot_crop        : bottom half cropped to white border, or None
        top_text        : OCR result from top half, or ""
        bot_text        : OCR result from bottom half, or ""
    """
    global _last_sign_roi, _last_top_crop, _last_bot_crop
    global _last_top_text, _last_bot_text

    annotated = frame.copy()
    mask      = _blue_mask(frame)
    rect      = _find_best_sign(frame, mask)

    if rect is None:
        _last_sign_roi = None
        _last_top_crop = None
        _last_bot_crop = None
        _last_top_text = ""
        _last_bot_text = ""
        if _result_callback:
            _result_callback(None, None, None, "", "")
        return annotated, None, None, None, "", ""

    x, y, w, h = rect

    cv2.rectangle(annotated, (x, y), (x + w, y + h), ANNO_COLOR, 2)
    cv2.putText(annotated, "SIGN", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ANNO_COLOR, 2)

    sign_roi = _extract_interior(frame, rect)
    top_crop, bot_crop, top_text, bot_text = _read_sign(sign_roi)

    cv2.putText(annotated, top_text, (x, y + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ANNO_COLOR, 1)
    cv2.putText(annotated, bot_text, (x, y + h + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ANNO_COLOR, 1)

    _last_sign_roi = sign_roi
    _last_top_crop = top_crop
    _last_bot_crop = bot_crop
    _last_top_text = top_text
    _last_bot_text = bot_text

    if _result_callback:
        _result_callback(sign_roi, top_crop, bot_crop, top_text, bot_text)

    return annotated, sign_roi, top_crop, bot_crop, top_text, bot_text