#!/usr/bin/env python3
"""
step_calibrator.py
------------------
GUI tool for step-by-step waypoint calibration.

Workflow per step:
  1. Choose a direction and enter a duration (seconds)
  2. Click Play → robot moves
  3. Click Rewind → robot teleports back to step start position
  4. Adjust duration and Play again as needed
  5. Click Confirm when the move is correct → saves step, advances

Other controls:
  - Delete Last       : removes last confirmed step, rewinds to its start
  - Rewind to Sel.    : teleports to selected step's start, removes that
                        step and everything after (for re-calibration)
  - Replay from Sel.  : teleports to selected step's start, plays all
                        steps from there onward (non-destructive)
  - Replay All        : teleports to step 1 start, plays everything
  - Print MOVEMENTS   : prints the MOVEMENTS list to the log and copies
                        it to the clipboard

Usage:
    rosrun <your_package> step_calibrator.py
"""

import math
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState

# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

CMD_VEL_TOPIC  = "/B1/cmd_vel"
MODEL_NAME     = "B1"
FORWARD_SPEED  = 0.3   # m/s
TURN_SPEED     = 2.0   # rad/s
PUBLISH_HZ     = 10

# vx, vy, wz are now specified directly as signed floats in the GUI


# ------------------------------------------------------------------ #
# Data helpers                                                         #
# ------------------------------------------------------------------ #

def quaternion_to_yaw_deg(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.degrees(math.atan2(siny, cosy))


class Pose:
    """Stores the full 7-DOF Gazebo pose so rewind is exact."""
    def __init__(self, x=0.0, y=0.0, z=0.0,
                 qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.x  = x;  self.y  = y;  self.z  = z
        self.qx = qx; self.qy = qy; self.qz = qz; self.qw = qw

    @property
    def yaw_deg(self):
        siny = 2.0 * (self.qw * self.qz + self.qx * self.qy)
        cosy = 1.0 - 2.0 * (self.qy * self.qy + self.qz * self.qz)
        return math.degrees(math.atan2(siny, cosy))

    def __str__(self):
        return f"x={self.x:7.3f}  y={self.y:7.3f}  yaw={self.yaw_deg:7.1f}°"


class Step:
    def __init__(self, vx, vy, wz, duration, start_pose):
        self.vx         = vx
        self.vy         = vy
        self.wz         = wz
        self.duration   = duration
        self.start_pose = start_pose
        self.end_pose   = None   # filled on confirm

    def list_label(self, index):
        return (f"{index:2}.  vx={self.vx:+.2f} vy={self.vy:+.2f}"
                f" wz={self.wz:+.2f}  {self.duration:.3f}s")


# ================================================================== #
# Main application                                                    #
# ================================================================== #

class StepCalibratorApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Step Calibrator")
        self.root.geometry("940x680")
        self.root.resizable(True, True)

        # ROS handles (set once ROS is ready)
        self.pub       = None
        self.ros_ready = threading.Event()

        # State
        self.confirmed_steps      = []     # list of Step
        self.current_start_pose   = None   # Pose — start of the step being calibrated
        self.busy                 = False

        # Command vars (set before _build_ui so widgets can bind them)
        self.vx_var  = tk.DoubleVar(value=0.3)
        self.vy_var  = tk.DoubleVar(value=0.0)
        self.wz_var  = tk.DoubleVar(value=0.0)

        self._build_ui()
        self._start_ros()

    # ---------------------------------------------------------------- #
    # ROS setup                                                         #
    # ---------------------------------------------------------------- #

    def _start_ros(self):
        # Publisher created here — init_node already called in main thread
        self.pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)

        def _background():
            rospy.sleep(1.0)
            self.ros_ready.set()
            pose = self._get_gazebo_pose()
            if pose:
                self.current_start_pose = pose
                self.root.after(0, self._refresh_pose_labels)
            self.root.after(0, lambda: self._set_status("Ready."))
            self.log("ROS ready. Robot start pose captured.")
            rospy.spin()

        threading.Thread(target=_background, daemon=True).start()

    def _get_gazebo_pose(self):
        try:
            rospy.wait_for_service("/gazebo/get_model_state", timeout=5.0)
            svc  = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
            resp = svc(MODEL_NAME, "world")
            p    = resp.pose.position
            q    = resp.pose.orientation
            return Pose(p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        except Exception as e:
            self.log(f"get_pose error: {e}")
            return None

    def _set_gazebo_pose(self, pose: Pose):
        msg = ModelState()
        msg.model_name        = MODEL_NAME
        msg.pose.position.x   = pose.x
        msg.pose.position.y   = pose.y
        msg.pose.position.z   = pose.z
        msg.pose.orientation.x = pose.qx
        msg.pose.orientation.y = pose.qy
        msg.pose.orientation.z = pose.qz
        msg.pose.orientation.w = pose.qw
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state(msg)
            return True
        except rospy.ServiceException:
            self.log("set_pose: service call failed")
            return False

    def _execute_move(self, vx, vy, wz, duration):
        """Run one move. Must be called from a background thread."""
        cmd           = Twist()
        cmd.linear.x  = vx
        cmd.linear.y  = vy
        cmd.angular.z = wz
        deadline      = time.time() + duration
        rate          = rospy.Rate(PUBLISH_HZ)
        while time.time() < deadline and not rospy.is_shutdown():
            self.pub.publish(cmd)
            rate.sleep()
        self.pub.publish(Twist())

    # ---------------------------------------------------------------- #
    # UI build                                                          #
    # ---------------------------------------------------------------- #

    def _build_ui(self):
        self.root.configure(bg="#1e1e2e")

        # Colour palette
        BG      = "#1e1e2e"
        PANEL   = "#2a2a3e"
        ACCENT  = "#89b4fa"
        GREEN   = "#a6e3a1"
        ORANGE  = "#fab387"
        RED     = "#f38ba8"
        TEXT    = "#cdd6f4"
        MUTED   = "#6c7086"
        ENTRY   = "#313244"

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
                        fieldbackground=ENTRY,
                        background=ENTRY,
                        foreground=TEXT,
                        selectbackground=ENTRY,
                        selectforeground=TEXT)
        style.map("TCombobox", fieldbackground=[("readonly", ENTRY)])

        # ── top label ──
        tk.Label(self.root, text="Step Calibrator",
                 bg=BG, fg=ACCENT,
                 font=("Courier", 16, "bold")).pack(pady=(10, 4))

        # ── main pane ──
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # LEFT column ─────────────────────────────────────────────
        left = tk.Frame(main, bg=PANEL, bd=0, relief=tk.FLAT)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                  padx=(0, 6), pady=0)

        tk.Label(left, text="CONFIRMED STEPS",
                 bg=PANEL, fg=MUTED,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=10, pady=(8, 2))

        list_frame = tk.Frame(left, bg=PANEL)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))

        sb = tk.Scrollbar(list_frame, bg=PANEL)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.steps_list = tk.Listbox(
            list_frame,
            yscrollcommand=sb.set,
            bg=ENTRY, fg=TEXT,
            selectbackground=ACCENT, selectforeground="#1e1e2e",
            font=("Courier", 10),
            bd=0, highlightthickness=0,
            activestyle="none"
        )
        self.steps_list.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.steps_list.yview)
        self.steps_list.bind("<<ListboxSelect>>", self._on_step_selected)

        # left buttons
        def left_btn(parent, text, cmd, fg=TEXT):
            return tk.Button(parent, text=text, command=cmd,
                             bg=ENTRY, fg=fg, activebackground=ACCENT,
                             activeforeground="#1e1e2e",
                             font=("Courier", 9), relief=tk.FLAT,
                             bd=0, padx=8, pady=5, cursor="hand2")

        btn_row1 = tk.Frame(left, bg=PANEL)
        btn_row1.pack(fill=tk.X, padx=8, pady=2)
        self.btn_delete = left_btn(btn_row1, "✕  Delete Last", self.delete_last, RED)
        self.btn_delete.pack(fill=tk.X, pady=2)
        self.btn_update = left_btn(btn_row1, "✎  Update Selected", self.update_selected, ORANGE)
        self.btn_update.pack(fill=tk.X, pady=2)

        btn_row2 = tk.Frame(left, bg=PANEL)
        btn_row2.pack(fill=tk.X, padx=8, pady=2)
        self.btn_rewind_sel = left_btn(btn_row2, "↺  Rewind to Selected",
                                       self.rewind_to_selected, ORANGE)
        self.btn_rewind_sel.pack(fill=tk.X, pady=2)

        self.btn_replay_sel = left_btn(btn_row2, "▶  Replay from Selected",
                                       self.replay_from_selected, GREEN)
        self.btn_replay_sel.pack(fill=tk.X, pady=2)

        self.btn_replay_all = left_btn(btn_row2, "▶▶ Replay All (drive)",
                                       self.replay_all, ACCENT)

        self.btn_replay_chained = left_btn(btn_row2, "▶▶ Replay All (teleport)",
                                          self.replay_all_chained, MUTED)
        self.btn_replay_all.pack(fill=tk.X, pady=2)

        self.btn_export = left_btn(btn_row2, "⎘  Copy MOVEMENTS",
                                   self.export_movements, TEXT)
        self.btn_export.pack(fill=tk.X, pady=(2, 8))

        # RIGHT column ────────────────────────────────────────────
        right = tk.Frame(main, bg=PANEL, bd=0)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=0)

        tk.Label(right, text="CURRENT STEP",
                 bg=PANEL, fg=MUTED,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=10, pady=(8, 2))

        ctrl = tk.Frame(right, bg=PANEL)
        ctrl.pack(fill=tk.X, padx=12, pady=6)

        # Direction
        def labeled_entry(row, label, var, hint=""):
            tk.Label(ctrl, text=label, bg=PANEL, fg=MUTED,
                     font=("Courier", 9)).grid(row=row, column=0, sticky="w", pady=3)
            tk.Entry(ctrl, textvariable=var, width=10,
                     bg=ENTRY, fg=TEXT, insertbackground=TEXT,
                     relief=tk.FLAT, font=("Courier", 10)).grid(
                         row=row, column=1, sticky="w", padx=8)
            if hint:
                tk.Label(ctrl, text=hint, bg=PANEL, fg=MUTED,
                         font=("Courier", 8)).grid(row=row, column=2, sticky="w")

        labeled_entry(0, "vx  (m/s)", self.vx_var,  "+fwd  −bwd")
        labeled_entry(1, "vy  (m/s)", self.vy_var,  "+left −right")
        labeled_entry(2, "ωz  (r/s)", self.wz_var,  "+CCW  −CW")

        # Duration
        tk.Label(ctrl, text="Duration (s)", bg=PANEL, fg=MUTED,
                 font=("Courier", 9)).grid(row=3, column=0, sticky="w", pady=3)
        self.dur_var = tk.StringVar(value="1.0")
        tk.Entry(ctrl, textvariable=self.dur_var, width=10,
                 bg=ENTRY, fg=TEXT, insertbackground=TEXT,
                 relief=tk.FLAT, font=("Courier", 10)).grid(
                     row=3, column=1, sticky="w", padx=8)

        # Action buttons
        actions = tk.Frame(right, bg=PANEL)
        actions.pack(fill=tk.X, padx=12, pady=8)

        def action_btn(parent, text, cmd, color):
            return tk.Button(parent, text=text, command=cmd,
                             bg=color, fg="#1e1e2e",
                             activebackground=TEXT, activeforeground="#1e1e2e",
                             font=("Courier", 10, "bold"),
                             relief=tk.FLAT, bd=0, padx=12, pady=7,
                             cursor="hand2")

        self.btn_play    = action_btn(actions, "▶  Play",    self.play_step,    GREEN)
        self.btn_rewind  = action_btn(actions, "↺  Rewind",  self.rewind,       ORANGE)
        self.btn_confirm = action_btn(actions, "✓  Confirm", self.confirm_step, ACCENT)

        self.btn_play.pack(side=tk.LEFT, padx=3)
        self.btn_rewind.pack(side=tk.LEFT, padx=3)
        self.btn_confirm.pack(side=tk.LEFT, padx=3)

        # Pose display
        pose_frame = tk.Frame(right, bg=ENTRY, bd=0)
        pose_frame.pack(fill=tk.X, padx=12, pady=6)

        tk.Label(pose_frame, text="POSE INFO", bg=ENTRY, fg=MUTED,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=8, pady=(6,2))

        self.lbl_step_start = tk.Label(pose_frame, text="Step start :  —",
                                       bg=ENTRY, fg=MUTED,
                                       font=("Courier", 9), anchor="w")
        self.lbl_step_start.pack(fill=tk.X, padx=8, pady=1)

        self.lbl_current = tk.Label(pose_frame, text="Current    :  —",
                                    bg=ENTRY, fg=TEXT,
                                    font=("Courier", 9), anchor="w")
        self.lbl_current.pack(fill=tk.X, padx=8, pady=(1, 8))

        # Status bar
        self.lbl_status = tk.Label(right, text="Status: Initializing…",
                                   bg=PANEL, fg=MUTED,
                                   font=("Courier", 9, "italic"), anchor="w")
        self.lbl_status.pack(fill=tk.X, padx=12, pady=4)

        # ── log ──────────────────────────────────────────────────
        log_outer = tk.Frame(self.root, bg=BG)
        log_outer.pack(fill=tk.BOTH, padx=10, pady=(4, 8))

        tk.Label(log_outer, text="LOG", bg=BG, fg=MUTED,
                 font=("Courier", 8, "bold")).pack(anchor="w")

        self.log_box = scrolledtext.ScrolledText(
            log_outer, height=8,
            bg=ENTRY, fg=MUTED,
            font=("Courier", 9),
            state=tk.DISABLED, relief=tk.FLAT, bd=0
        )
        self.log_box.pack(fill=tk.BOTH, expand=True)

        # store colours for later use in set_busy
        self._colors = dict(GREEN=GREEN, ORANGE=ORANGE, ACCENT=ACCENT,
                            RED=RED, ENTRY=ENTRY, MUTED=MUTED)

    # ---------------------------------------------------------------- #
    # GUI helpers (all callable from any thread via root.after)         #
    # ---------------------------------------------------------------- #

    def log(self, msg):
        def _write():
            self.log_box.config(state=tk.NORMAL)
            self.log_box.insert(tk.END, f"  {msg}\n")
            self.log_box.see(tk.END)
            self.log_box.config(state=tk.DISABLED)
        self.root.after(0, _write)

    def _set_status(self, msg):
        self.root.after(0, lambda: self.lbl_status.config(text=f"Status: {msg}"))

    def _set_busy(self, busy: bool):
        self.busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        c     = self._colors

        def _update():
            for btn in (self.btn_play, self.btn_rewind, self.btn_confirm,
                        self.btn_delete, self.btn_update, self.btn_rewind_sel,
                        self.btn_replay_sel, self.btn_replay_all,
                        self.btn_replay_chained, self.btn_export):
                btn.config(state=state)
            # dim text when busy
            fg = c["MUTED"] if busy else c["GREEN"]
            self.btn_play.config(fg="#1e1e2e" if not busy else c["MUTED"])

        self.root.after(0, _update)

    def _refresh_steps_list(self):
        def _update():
            self.steps_list.delete(0, tk.END)
            for i, step in enumerate(self.confirmed_steps):
                self.steps_list.insert(tk.END, step.list_label(i + 1))
        self.root.after(0, _update)

    def _refresh_pose_labels(self):
        """Fetch pose in a background thread then update labels in main thread."""
        def _fetch():
            pose = self._get_gazebo_pose()
            def _update():
                if self.current_start_pose:
                    self.lbl_step_start.config(
                        text=f"Step start :  {self.current_start_pose}")
                if pose:
                    self.lbl_current.config(text=f"Current    :  {pose}")
            self.root.after(0, _update)
        threading.Thread(target=_fetch, daemon=True).start()

    def _check_ros(self):
        if not self.ros_ready.is_set():
            messagebox.showwarning("Not Ready", "ROS is not ready yet. Please wait.")
            return False
        return True

    def _parse_duration(self):
        try:
            d = float(self.dur_var.get())
            if d <= 0:
                raise ValueError
            return d
        except ValueError:
            messagebox.showerror("Error", "Duration must be a positive number.")
            return None

    # ---------------------------------------------------------------- #
    # Background task runner                                            #
    # ---------------------------------------------------------------- #

    def _run_in_thread(self, fn):
        """Execute fn() in a daemon thread, guarded by busy flag."""
        if self.busy:
            return
        if not self._check_ros():
            return

        def _wrapper():
            self._set_busy(True)
            try:
                fn()
            finally:
                self._set_busy(False)
                self.root.after(0, self._refresh_pose_labels)

        threading.Thread(target=_wrapper, daemon=True).start()

    # ---------------------------------------------------------------- #
    # Actions                                                           #
    # ---------------------------------------------------------------- #

    def play_step(self):
        duration = self._parse_duration()
        if duration is None:
            return
        vx = self.vx_var.get()
        vy = self.vy_var.get()
        wz = self.wz_var.get()

        def _do():
            self._set_status(
                f"Playing vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}  {duration:.3f}s …")
            self.log(f"▶  vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}  {duration:.3f}s")
            self._execute_move(vx, vy, wz, duration)
            self._set_status("Done. Rewind to retry or Confirm to save.")

        self._run_in_thread(_do)

    def rewind(self):
        if not self.current_start_pose:
            messagebox.showinfo("Info", "No start pose recorded yet.")
            return

        def _do():
            self._set_status("Rewinding …")
            self.log(f"↺  Rewind → {self.current_start_pose}")
            self._set_gazebo_pose(self.current_start_pose)
            rospy.sleep(0.3)
            self._set_status("Rewound. Adjust duration and Play again.")

        self._run_in_thread(_do)

    def confirm_step(self):
        duration = self._parse_duration()
        if duration is None:
            return
        vx = self.vx_var.get()
        vy = self.vy_var.get()
        wz = self.wz_var.get()

        def _do():
            end_pose = self._get_gazebo_pose()
            step     = Step(vx, vy, wz, duration, self.current_start_pose)
            step.end_pose = end_pose
            self.confirmed_steps.append(step)
            self.current_start_pose = end_pose
            n = len(self.confirmed_steps)
            self.log(f"✓  Step {n}: vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}  {duration:.3f}s")
            self._refresh_steps_list()
            self._set_status(f"Step {n} confirmed. Set up next step.")

        self._run_in_thread(_do)

    def delete_last(self):
        if not self.confirmed_steps:
            return

        def _do():
            step = self.confirmed_steps.pop()
            self.current_start_pose = step.start_pose
            self._set_gazebo_pose(step.start_pose)
            rospy.sleep(0.3)
            self.log(f"✕  Deleted step {len(self.confirmed_steps)+1}. "
                     f"Rewound to its start.")
            self._refresh_steps_list()
            self._set_status("Last step deleted.")

        self._run_in_thread(_do)

    def rewind_to_selected(self):
        sel = self.steps_list.curselection()
        if not sel:
            messagebox.showinfo("Info", "Select a step from the list first.")
            return
        idx = sel[0]

        def _do():
            step = self.confirmed_steps[idx]
            # Keep only steps before the selected one
            self.confirmed_steps[:] = self.confirmed_steps[:idx]
            self.current_start_pose = step.start_pose
            self._set_gazebo_pose(step.start_pose)
            rospy.sleep(0.3)
            self.log(f"↺  Rewound to start of step {idx+1}. "
                     f"Steps {idx+1}+ removed.")
            self._refresh_steps_list()
            self._set_status(f"At step {idx+1} start. Re-calibrate from here.")

        self._run_in_thread(_do)

    def replay_from_selected(self):
        sel = self.steps_list.curselection()
        if not sel:
            messagebox.showinfo("Info", "Select a step from the list first.")
            return
        idx        = sel[0]
        start_pose = self.confirmed_steps[idx].start_pose
        steps      = self.confirmed_steps[idx:]
        self._replay(steps, start_pose, f"from step {idx+1}",
                     teleport_each_step=False)

    def replay_all(self):
        if not self.confirmed_steps:
            return
        self._replay(self.confirmed_steps,
                     self.confirmed_steps[0].start_pose, "all",
                     teleport_each_step=False)

    def replay_all_chained(self):
        """Replay with teleport before each step — for per-step verification."""
        if not self.confirmed_steps:
            return
        self._replay(self.confirmed_steps,
                     self.confirmed_steps[0].start_pose, "all (teleport)",
                     teleport_each_step=True)

    def _replay(self, steps, start_pose, label, teleport_each_step=True):
        """
        Replay a sequence of steps.

        teleport_each_step=True:  teleport to each step's saved start_pose
                                   before executing. Guarantees each step
                                   starts from the exact calibration position.
        teleport_each_step=False: chain steps physically (no teleport between).
        """
        def _do():
            self.log(f"▶▶ Replaying {label} ({len(steps)} steps)"
                     f"  [{'teleport' if teleport_each_step else 'chained'}] …")
            self._set_gazebo_pose(start_pose)
            rospy.sleep(0.5)
            for i, step in enumerate(steps):
                if rospy.is_shutdown():
                    break
                if teleport_each_step and step.start_pose is not None:
                    self._set_gazebo_pose(step.start_pose)
                    rospy.sleep(0.2)
                self._set_status(
                    f"Replay {i+1}/{len(steps)}: vx={step.vx:+.2f} vy={step.vy:+.2f} wz={step.wz:+.2f}  {step.duration:.3f}s")
                self.log(f"   [{i+1}] vx={step.vx:+.2f} vy={step.vy:+.2f} wz={step.wz:+.2f}  {step.duration:.3f}s")
                self._execute_move(step.vx, step.vy, step.wz, step.duration)
                rospy.sleep(0.1)
            self.root.after(0, self._refresh_pose_labels)
            self.log("▶▶ Replay complete.")
            self._set_status("Replay complete.")

        self._run_in_thread(_do)

    def _on_step_selected(self, event):
        """Load the selected step's values into the input fields."""
        sel = self.steps_list.curselection()
        if not sel:
            return
        step = self.confirmed_steps[sel[0]]
        self.vx_var.set(round(step.vx, 4))
        self.vy_var.set(round(step.vy, 4))
        self.wz_var.set(round(step.wz, 4))
        self.dur_var.set(f"{step.duration:.4f}")
        self.log(f"  Loaded step {sel[0]+1} into fields for editing.")

    def update_selected(self):
        """Replace the selected step's command values with the current fields."""
        sel = self.steps_list.curselection()
        if not sel:
            messagebox.showinfo("Info", "Select a step from the list first.")
            return
        duration = self._parse_duration()
        if duration is None:
            return
        idx = sel[0]
        vx  = self.vx_var.get()
        vy  = self.vy_var.get()
        wz  = self.wz_var.get()

        step = self.confirmed_steps[idx]
        step.vx       = vx
        step.vy       = vy
        step.wz       = wz
        step.duration = duration
        # start_pose and end_pose are preserved unchanged

        self.log(f"  ✎ Step {idx+1} updated: vx={vx:+.2f} vy={vy:+.2f}"
                 f" wz={wz:+.2f}  {duration:.3f}s")
        self._refresh_steps_list()
        # Re-select the same item after list refresh
        self.root.after(50, lambda: (
            self.steps_list.selection_set(idx),
            self.steps_list.see(idx)
        ))
        self._set_status(f"Step {idx+1} updated.")

    def export_movements(self):
        if not self.confirmed_steps:
            self.log("No steps to export.")
            return
        lines = ["MOVEMENTS = ["]
        lines.append("    # (vx m/s, vy m/s, wz rad/s, duration s)")
        for step in self.confirmed_steps:
            lines.append(
                f"    ({step.vx:+.4f}, {step.vy:+.4f},"
                f" {step.wz:+.4f}, {step.duration:.4f}),"
            )
        lines.append("]")
        output = "\n".join(lines)

        self.log("─" * 44)
        for line in lines:
            self.log(line)
        self.log("─" * 44)

        self.root.clipboard_clear()
        self.root.clipboard_append(output)
        self.log("(Copied to clipboard)")


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # rospy.init_node must be called from the main thread
    # (Python signal handling requires it)
    rospy.init_node("step_calibrator", anonymous=True)
    root = tk.Tk()
    app  = StepCalibratorApp(root)
    root.mainloop()