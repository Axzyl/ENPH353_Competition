#!/usr/bin/env python3
"""
camera_feed.py
--------------
Camera feed reader with perspective correction and a live tkinter UI.

Topics:
  /B1/robot/camera_fl/image_raw   — left  camera (25° outward)
  /B1/rrbot/camera1/image_raw     — center camera (straight)
  /B1/robot/camera_fr/image_raw   — right camera (25° outward)

API:
    import camera_feed

    camera_feed.init()              # call once after rospy.init_node
    camera_feed.start("center")     # begin processing + updating UI
    camera_feed.stop()              # freeze UI (window stays open)
    camera_feed.switch("left")      # switch to another camera

Future processing hooks:
    Register a processor function with camera_feed.add_processor(fn).
    fn receives a corrected BGR frame and returns an annotated frame.
    Multiple processors are chained in registration order.
"""

import base64
import math
import threading
import time
import tkinter as tk

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# ------------------------------------------------------------------ #
# Camera topics                                                        #
# ------------------------------------------------------------------ #

CAMERAS = {
    "left":   "/B1/robot/camera_fl/image_raw",
    "center": "/B1/rrbot/camera1/image_raw",
    "right":  "/B1/robot/camera_fr/image_raw",
}

# Outward angle of left/right cameras in degrees
SIDE_ANGLE_DEG = 25.0

# ------------------------------------------------------------------ #
# Homography settings — tune these if correction looks wrong           #
# ------------------------------------------------------------------ #

# Set to False to disable perspective correction entirely
ENABLE_HOMOGRAPHY = True

# Assumed horizontal FOV of the side cameras in degrees.
# Increase if correction is too strong, decrease if too weak.
CAMERA_FOV_DEG = 60.0

# ------------------------------------------------------------------ #

# Display size for the UI canvas
DISPLAY_W = 640
DISPLAY_H = 480

# Sign panel dimensions (right of camera feed)
SIGN_PANEL_W = 280
SIGN_IMG_H   = 140

# ------------------------------------------------------------------ #
# Internal state                                                       #
# ------------------------------------------------------------------ #

bridge      = CvBridge()
_subs       = {}           # active rospy subscribers
_latest     = {}           # latest raw frame per camera
_homography = {}           # precomputed homography per camera
_processors = []           # list of frame-processing functions
_active_cam = None         # currently processing camera name
_stop_event = threading.Event()   # set → stop; clear → run
_thread     = None
_inited     = False

# Tkinter handles
_root         = None
_canvas       = None
_lbl_cam      = None
_lbl_status   = None
_tk_thread    = None

# Sign panel handles
_sign_canvas  = None
_lbl_top_text = None
_lbl_bot_text = None


# ================================================================== #
# HOMOGRAPHY                                                          #
# ================================================================== #

def _build_homography(camera_angle_deg, img_w, img_h):
    """
    Build a homography matrix to correct perspective distortion caused
    by the camera being yawed `camera_angle_deg` degrees outward.

    camera_angle_deg > 0 : camera is rotated to the right (right cam)
    camera_angle_deg < 0 : camera is rotated to the left  (left cam)

    The correction applies the INVERSE rotation: H = K * R_y(-α) * K^-1
    This maps pixel positions from the distorted camera image to where
    they would appear if the camera was pointing straight.
    """
    f  = (img_w / 2.0) / math.tan(math.radians(CAMERA_FOV_DEG / 2.0))
    cx = img_w / 2.0
    cy = img_h / 2.0

    K = np.array([[f,  0, cx],
                  [0,  f, cy],
                  [0,  0,  1]], dtype=np.float64)

    # Apply NEGATIVE of camera angle to undo the rotation
    theta = math.radians(-camera_angle_deg)
    Ry = np.array([[ math.cos(theta), 0, math.sin(theta)],
                   [ 0,               1, 0              ],
                   [-math.sin(theta), 0, math.cos(theta)]], dtype=np.float64)

    H = K @ Ry @ np.linalg.inv(K)
    return H


def _precompute_homographies(img_w=640, img_h=480):
    """Precompute homography matrices for left/right cameras."""
    global _homography
    # Left camera is rotated outward to the left (-25°)
    _homography["left"]   = _build_homography(-SIDE_ANGLE_DEG, img_w, img_h)
    # Right camera is rotated outward to the right (+25°)
    _homography["right"]  = _build_homography(+SIDE_ANGLE_DEG, img_w, img_h)
    _homography["center"] = None   # no correction needed


def _apply_homography(frame, camera):
    """Apply perspective correction if needed for this camera."""
    if not ENABLE_HOMOGRAPHY:
        return frame
    H = _homography.get(camera)
    if H is None:
        return frame
    h, w = frame.shape[:2]
    # Recompute if image size differs from precomputed size
    if h != 480 or w != 640:
        _precompute_homographies(w, h)
        H = _homography.get(camera)
    return cv2.warpPerspective(frame, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


# ================================================================== #
# ROSPY SUBSCRIBERS                                                   #
# ================================================================== #

def _make_callback(name):
    def cb(msg):
        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            _latest[name] = img
        except Exception as e:
            rospy.logwarn("camera_feed: decode error on %s: %s", name, e)
    return cb


def _ensure_subscribed(camera):
    if camera not in _subs:
        topic = CAMERAS[camera]
        _subs[camera] = rospy.Subscriber(topic, Image, _make_callback(camera))
        rospy.loginfo("camera_feed: subscribed to %s", topic)


# ================================================================== #
# TKINTER UI                                                          #
# ================================================================== #

def _build_ui():
    global _root, _canvas, _lbl_cam, _lbl_status
    global _sign_canvas, _lbl_top_text, _lbl_bot_text

    _root = tk.Tk()
    _root.title("Camera Feed")
    _root.configure(bg="#0d0d0d")
    _root.resizable(False, False)

    # Header
    header = tk.Frame(_root, bg="#0d0d0d")
    header.pack(fill=tk.X, padx=12, pady=(10, 4))

    tk.Label(header, text="CAMERA FEED",
             bg="#0d0d0d", fg="#e0e0e0",
             font=("Courier", 13, "bold")).pack(side=tk.LEFT)

    _lbl_cam = tk.Label(header, text="[ none ]",
                        bg="#0d0d0d", fg="#5af78e",
                        font=("Courier", 11))
    _lbl_cam.pack(side=tk.LEFT, padx=12)

    _lbl_status = tk.Label(header, text="● IDLE",
                           bg="#0d0d0d", fg="#666",
                           font=("Courier", 10))
    _lbl_status.pack(side=tk.RIGHT)

    # Main content row: camera feed + sign panel
    content_row = tk.Frame(_root, bg="#0d0d0d")
    content_row.pack(padx=12, pady=(0, 12))

    # Camera canvas
    cam_frame = tk.Frame(content_row, bg="#111", bd=0)
    cam_frame.pack(side=tk.LEFT)

    _canvas = tk.Canvas(cam_frame,
                        width=DISPLAY_W, height=DISPLAY_H,
                        bg="#111", highlightthickness=1,
                        highlightbackground="#333")
    _canvas.pack()
    _canvas.create_text(DISPLAY_W // 2, DISPLAY_H // 2,
                        text="NO FEED", fill="#333",
                        font=("Courier", 18, "bold"),
                        tags="placeholder")

    # Sign panel (right side)
    sign_panel = tk.Frame(content_row, bg="#111",
                          width=SIGN_PANEL_W, height=DISPLAY_H,
                          highlightthickness=1,
                          highlightbackground="#333")
    sign_panel.pack(side=tk.LEFT, padx=(8, 0), fill=tk.Y)
    sign_panel.pack_propagate(False)

    tk.Label(sign_panel, text="SIGN", bg="#111", fg="#555",
             font=("Courier", 9, "bold")).pack(pady=(8, 4))

    _sign_canvas = tk.Canvas(sign_panel,
                             width=SIGN_PANEL_W - 16,
                             height=SIGN_IMG_H,
                             bg="#0d0d0d", highlightthickness=0)
    _sign_canvas.pack(padx=8)
    _sign_canvas.create_text((SIGN_PANEL_W - 16) // 2, SIGN_IMG_H // 2,
                             text="—", fill="#333",
                             font=("Courier", 14))

    tk.Label(sign_panel, text="TOP", bg="#111", fg="#555",
             font=("Courier", 8)).pack(anchor="w", padx=8, pady=(10, 0))
    _lbl_top_text = tk.Label(sign_panel, text="",
                             bg="#111", fg="#5af78e",
                             font=("Courier", 11, "bold"),
                             wraplength=SIGN_PANEL_W - 16,
                             justify="left")
    _lbl_top_text.pack(anchor="w", padx=8)

    tk.Label(sign_panel, text="BOTTOM", bg="#111", fg="#555",
             font=("Courier", 8)).pack(anchor="w", padx=8, pady=(8, 0))
    _lbl_bot_text = tk.Label(sign_panel, text="",
                             bg="#111", fg="#5af78e",
                             font=("Courier", 11, "bold"),
                             wraplength=SIGN_PANEL_W - 16,
                             justify="left")
    _lbl_bot_text.pack(anchor="w", padx=8)

    _root.protocol("WM_DELETE_WINDOW", _on_close)


def _on_close():
    """Keep window open — just hide it. Fully close only on script exit."""
    _root.withdraw()


def _tk_loop():
    """Tkinter main loop running in its own thread."""
    _build_ui()
    _root.mainloop()


def _update_canvas(bgr_frame):
    """Push a BGR frame to the tkinter canvas (called from processing thread)."""
    if _root is None or not _root.winfo_exists():
        return
    try:
        resized = cv2.resize(bgr_frame, (DISPLAY_W, DISPLAY_H),
                             interpolation=cv2.INTER_LINEAR)
        _, buf = cv2.imencode(".png", resized)
        b64    = base64.b64encode(buf.tobytes())

        def _draw():
            if _canvas is None:
                return
            img = tk.PhotoImage(data=b64)
            _canvas.delete("all")
            _canvas.create_image(0, 0, anchor="nw", image=img)
            _canvas._img_ref = img   # prevent GC

        _root.after(0, _draw)
    except Exception as e:
        rospy.logwarn("camera_feed: canvas update error: %s", e)


def _set_ui_status(camera, active):
    if _root is None:
        return
    def _update():
        if _lbl_cam:
            _lbl_cam.config(text=f"[ {camera} ]" if camera else "[ none ]")
        if _lbl_status:
            if active:
                _lbl_status.config(text="● LIVE", fg="#5af78e")
            else:
                _lbl_status.config(text="● IDLE", fg="#666")
    _root.after(0, _update)


# ================================================================== #
# PROCESSING THREAD                                                   #
# ================================================================== #

def _processing_loop(camera):
    """Main processing loop — runs in a daemon thread."""
    rospy.loginfo("camera_feed: processing started [%s]", camera)
    rate = rospy.Rate(30)

    while not _stop_event.is_set() and not rospy.is_shutdown():
        frame = _latest.get(camera)
        if frame is None:
            rate.sleep()
            continue

        # 1. Perspective correction
        corrected = _apply_homography(frame.copy(), camera)

        # 2. Run registered processors in order
        output = corrected
        for fn in list(_processors):
            try:
                result = fn(output)
                if result is None:
                    pass
                elif isinstance(result, tuple):
                    # Processors that return (frame, ...) — take first element as frame
                    if len(result) > 0 and isinstance(result[0], np.ndarray):
                        output = result[0]
                else:
                    output = result
            except Exception as e:
                rospy.logwarn("camera_feed: processor error: %s", e)

        # 3. Push to UI
        _update_canvas(output)
        rate.sleep()

    rospy.loginfo("camera_feed: processing stopped [%s]", camera)


# ================================================================== #
# PUBLIC API                                                          #
# ================================================================== #

def update_sign_panel(sign_roi, top_crop, bot_crop, top_text, bot_text):
    """
    Update the sign panel in the UI with a new sign image and text.
    Safe to call from any thread.

    Args:
        sign_roi : BGR numpy array of the extracted sign, or None
        top_crop : BGR numpy array of the cropped top half, or None
        bot_crop : BGR numpy array of the cropped bottom half, or None
        top_text : string for the top-half OCR result
        bot_text : string for the bottom-half OCR result
    """
    if _root is None:
        return

    def _draw():
        # Update text labels
        if _lbl_top_text:
            _lbl_top_text.config(text=top_text or "")
        if _lbl_bot_text:
            _lbl_bot_text.config(text=bot_text or "")

        # Update sign image canvas
        if _sign_canvas is None:
            return
        if sign_roi is None:
            _sign_canvas.delete("all")
            _sign_canvas.create_text(
                (SIGN_PANEL_W - 16) // 2, SIGN_IMG_H // 2,
                text="—", fill="#333", font=("Courier", 14))
            return
        try:
            import base64
            h, w = sign_roi.shape[:2]
            target_w = SIGN_PANEL_W - 16
            target_h = SIGN_IMG_H
            scale    = min(target_w / max(w, 1), target_h / max(h, 1))
            new_w    = max(1, int(w * scale))
            new_h    = max(1, int(h * scale))
            resized  = cv2.resize(sign_roi, (new_w, new_h))
            _, buf   = cv2.imencode(".png", resized)
            b64      = base64.b64encode(buf.tobytes())
            img      = tk.PhotoImage(data=b64)
            _sign_canvas.delete("all")
            # Centre the image
            ox = (target_w - new_w) // 2
            oy = (target_h - new_h) // 2
            _sign_canvas.create_image(ox, oy, anchor="nw", image=img)
            _sign_canvas._img_ref = img
        except Exception as e:
            pass

    _root.after(0, _draw)


def init():
    """
    Initialise the camera feed module. Call once after rospy.init_node().
    Launches the tkinter window in a background thread and automatically
    wires up sign_reader so the UI updates without any extra setup.
    """
    global _inited, _tk_thread
    if _inited:
        return

    # Precompute homographies with assumed image size
    _precompute_homographies(640, 480)

    # Launch tkinter in its own thread
    _tk_thread = threading.Thread(target=_tk_loop, daemon=True)
    _tk_thread.start()
    time.sleep(0.5)   # let window initialise

    # Auto-wire sign reader — updates sign panel without any main.py setup
    try:
        import sign_reader
        sign_reader.set_result_callback(update_sign_panel)
        add_processor(sign_reader.process_frame)
        rospy.loginfo("camera_feed: sign_reader registered automatically.")
    except ImportError:
        rospy.logwarn("camera_feed: sign_reader not found — sign panel disabled.")

    _inited = True
    rospy.loginfo("camera_feed: initialised.")


def start(camera):
    """
    Start processing the given camera and updating the UI feed.

    Args:
        camera : "left", "center", or "right"
    """
    global _active_cam, _thread

    if camera not in CAMERAS:
        raise ValueError(f"Unknown camera '{camera}'. Choose from: {list(CAMERAS)}")

    # Stop existing thread if running
    _stop_event.set()
    if _thread and _thread.is_alive():
        _thread.join(timeout=2.0)
    _stop_event.clear()

    _ensure_subscribed(camera)
    _active_cam = camera
    _set_ui_status(camera, active=True)

    _thread = threading.Thread(target=_processing_loop,
                               args=(camera,), daemon=True)
    _thread.start()
    rospy.loginfo("camera_feed: started [%s]", camera)


def stop():
    """
    Stop updating the UI feed. The window stays open showing the last frame.
    """
    _stop_event.set()
    _set_ui_status(_active_cam or "none", active=False)
    rospy.loginfo("camera_feed: stopped.")


def switch(camera):
    """
    Switch to a different camera. Stops the current feed and starts the new one.

    Args:
        camera : "left", "center", or "right"
    """
    rospy.loginfo("camera_feed: switching to [%s]", camera)
    start(camera)   # start() handles stopping the previous thread


def add_processor(fn):
    """
    Register a processing function to be applied to each frame.

    fn(frame: np.ndarray) -> np.ndarray
      Receives a perspective-corrected BGR frame.
      Returns an annotated/processed BGR frame (or None to pass through).

    Processors are chained in registration order. Example:

        def my_ocr(frame):
            # ... run OCR, draw bounding boxes ...
            return annotated_frame

        camera_feed.add_processor(my_ocr)
    """
    _processors.append(fn)
    rospy.loginfo("camera_feed: processor registered (%d total)", len(_processors))


def remove_processor(fn):
    """Remove a previously registered processor function."""
    if fn in _processors:
        _processors.remove(fn)


def get_active_camera():
    """Return the name of the currently active camera, or None."""
    return _active_cam if not _stop_event.is_set() else None


# ------------------------------------------------------------------ #
# Standalone test                                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    rospy.init_node("camera_feed_test", anonymous=True)
    init()
    rospy.sleep(1.0)

    print("Starting center camera...")
    start("center")
    rospy.sleep(5.0)

    print("Switching to left camera...")
    switch("left")
    rospy.sleep(5.0)

    print("Switching to right camera...")
    switch("right")
    rospy.sleep(5.0)

    print("Stopping feed...")
    stop()
    rospy.sleep(3.0)

    rospy.loginfo("Done.")