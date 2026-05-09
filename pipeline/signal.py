"""
signal.py
Frame differencing, optical flow, camera motion compensation.
"""
import cv2
import numpy as np


def frame_diff(f1_gray, f2_gray, thresh=30):
    """Absolute frame difference → binary mask"""
    diff = cv2.absdiff(f1_gray, f2_gray)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    return mask


def optical_flow_farneback(f1_gray, f2_gray):
    """Dense optical flow → (H,W,2) flow array"""
    return cv2.calcOpticalFlowFarneback(
        f1_gray,
        f2_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def flow_to_hsv(frame_bgr, flow):
    """Visualise optical flow as colour-coded overlay"""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(frame_bgr)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def stabilise(f1_gray, f2_gray):
    """ORB + homography camera motion compensation. Returns warped f2."""
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(f1_gray, None)
    kp2, des2 = orb.detectAndCompute(f2_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4:
        return f2_gray
    matches = sorted(
        cv2.BFMatcher(cv2.NORM_HAMMING, True).match(des1, des2),
        key=lambda x: x.distance,
    )[:50]
    if len(matches) < 4:
        return f2_gray
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        return f2_gray
    h, w = f1_gray.shape
    return cv2.warpPerspective(f2_gray, H, (w, h))
