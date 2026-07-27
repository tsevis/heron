"""Stylized skeletal proxy for the Fluoroscope (CLAUDE.md §1.1, §3.2, §2.8).

MediaPipe pose keypoints drive a procedural bone drawing — skull, spine, clavicles,
a suggestion of ribs, and limb bones — rendered as bright soft structures inside
the body. This is deliberately **stylized-anatomical, never diagnostic** (§2.8):
it evokes a luminous internal core, not a real radiograph. Degrades to None if
MediaPipe is unavailable or no pose is found, so the fluoroscope still renders
its translucent-body look without it.
"""

from __future__ import annotations

import cv2
import numpy as np

from heron.color import srgb

# BlazePose (33 landmarks) indices we use
_NOSE = 0
_L_EAR, _R_EAR = 7, 8
_L_SHO, _R_SHO = 11, 12
_L_ELB, _R_ELB = 13, 14
_L_WRI, _R_WRI = 15, 16
_L_HIP, _R_HIP = 23, 24
_L_KNE, _R_KNE = 25, 26
_L_ANK, _R_ANK = 27, 28


def _detect_pose(linear_rgb: np.ndarray):
    """Return (landmarks_xy (33,2) px, visibility (33,)) or None."""
    try:
        import mediapipe as mp
    except ImportError:
        return None
    h, w = linear_rgb.shape[:2]
    rgb8 = np.clip(srgb.linear_to_srgb(linear_rgb) * 255, 0, 255).astype(np.uint8)
    try:
        with mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=2, min_detection_confidence=0.3
        ) as pose:
            res = pose.process(rgb8)
    except Exception:
        return None
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks.landmark
    xy = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    vis = np.array([p.visibility for p in lm], dtype=np.float32)
    return xy, vis


def skeleton_proxy(linear_rgb: np.ndarray) -> np.ndarray | None:
    """Return a soft [0,1] bone-glow map (H,W), or None if no pose is found."""
    det = _detect_pose(linear_rgb)
    if det is None:
        return None
    xy, vis = det
    return draw_skeleton(xy, vis, linear_rgb.shape[:2])


def draw_skeleton(xy: np.ndarray, vis: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    """Draw the stylized skeleton from (33,2) px landmarks + visibility.

    Separated from detection so the drawing is unit-testable without MediaPipe.
    """
    h, w = shape
    canvas = np.zeros((h, w), np.float32)

    def pt(i):
        return (int(round(xy[i, 0])), int(round(xy[i, 1])))

    def ok(*idx, thresh=0.3):
        return all(vis[i] > thresh for i in idx)

    # bone thickness scales with body size (shoulder span or image size)
    if ok(_L_SHO, _R_SHO):
        span = np.linalg.norm(xy[_L_SHO] - xy[_R_SHO])
    else:
        span = 0.25 * max(h, w)
    thick = max(2, int(span * 0.06))

    def bone(i, j):
        if ok(i, j):
            cv2.line(canvas, pt(i), pt(j), 1.0, thick, cv2.LINE_AA)

    # spine (mid-shoulders -> mid-hips)
    if ok(_L_SHO, _R_SHO, _L_HIP, _R_HIP):
        neck = (xy[_L_SHO] + xy[_R_SHO]) / 2
        pelvis = (xy[_L_HIP] + xy[_R_HIP]) / 2
        cv2.line(canvas, tuple(neck.astype(int)), tuple(pelvis.astype(int)), 1.0, thick, cv2.LINE_AA)
        # ribs: a few arcs suggested off the spine
        for f in (0.25, 0.45, 0.65):
            c = (neck * (1 - f) + pelvis * f).astype(int)
            axes = (int(span * 0.5 * (1.0 - f * 0.4)), int(span * 0.18))
            cv2.ellipse(canvas, tuple(c), axes, 0, 200, 340, 0.7, max(1, thick // 2), cv2.LINE_AA)

    bone(_L_SHO, _R_SHO)   # clavicles
    bone(_L_HIP, _R_HIP)   # pelvis
    bone(_L_SHO, _L_ELB); bone(_L_ELB, _L_WRI)  # left arm
    bone(_R_SHO, _R_ELB); bone(_R_ELB, _R_WRI)  # right arm
    bone(_L_HIP, _L_KNE); bone(_L_KNE, _L_ANK)  # left leg
    bone(_R_HIP, _R_KNE); bone(_R_KNE, _R_ANK)  # right leg

    # skull
    if ok(_NOSE) and ok(_L_EAR, _R_EAR):
        c = xy[_NOSE].astype(int)
        r = int(max(np.linalg.norm(xy[_L_EAR] - xy[_R_EAR]) * 0.7, span * 0.3))
        cv2.circle(canvas, tuple(c), r, 0.85, max(2, thick // 2), cv2.LINE_AA)

    # soften into a glow
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=max(1.0, thick * 0.4))
    m = float(canvas.max())
    return (canvas / m).astype(np.float32) if m > 1e-6 else None
