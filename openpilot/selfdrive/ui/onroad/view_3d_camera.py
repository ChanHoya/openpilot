"""Virtual windshield camera shared by the C3/C4 3D view.

Points use the calibrated device frame used by modelV2: x forward, y right,
z down, origin at the road camera. The road surface is about ``ground_z``
below the camera. The AR view keeps using the real camera transform; the 3D view
uses a single consistent slightly-wide windshield perspective so cruising leads,
close stopped leads, lanes, and road edges stay naturally framed.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CameraPose:
  back_m: float        # camera position behind the road camera
  height_m: float      # camera height above the road surface
  pitch_deg: float     # downward pitch
  focal_scale: float   # focal length as a fraction of min(width, 2 * height)
  horizon_frac: float  # horizon row as a fraction of the view height


# Single unified windshield perspective: slightly wide eye-level cabin view.
WINDSHIELD_POSE = CameraPose(back_m=1.8, height_m=1.65, pitch_deg=8.0, focal_scale=0.52, horizon_frac=0.36)
CRUISE_POSE = WINDSHIELD_POSE
WIDE_POSE = WINDSHIELD_POSE

CAR_WIDTH_M = 1.85
CAR_HEIGHT_M = 1.45
EGO_REAR_X_M = -2.6  # rear bumper relative to the road camera
ZOOM_TIME_CONSTANT_S = 0.45
MAX_YAW_DEG = 10.0


def zoom_target(v_ego: float, lead_d: float | None, curve_lat_m: float) -> float:
  """Unified windshield framing: no zoom oscillation."""
  return 0.0


class ChaseCamera:
  def __init__(self):
    self.zoom = 0.0
    self.yaw = 0.0
    self._w = 1.0
    self._h = 1.0
    self._x0 = 0.0
    self._y0 = 0.0
    self._ground_z = 1.22
    self._pose = WINDSHIELD_POSE

  def update(self, rect, ground_z: float, target: float, yaw_rad: float, dt: float) -> None:
    alpha = 1.0 if dt <= 0 else 1.0 - math.exp(-dt / ZOOM_TIME_CONSTANT_S)
    self.zoom += (float(np.clip(target, 0.0, 1.0)) - self.zoom) * alpha
    max_yaw = math.radians(MAX_YAW_DEG)
    self.yaw += (float(np.clip(yaw_rad, -max_yaw, max_yaw)) - self.yaw) * alpha
    self._x0, self._y0 = float(rect.x), float(rect.y)
    self._w, self._h = max(float(rect.width), 1.0), max(float(rect.height), 1.0)
    self._ground_z = float(ground_z)
    self._pose = WINDSHIELD_POSE

  def _focal_base(self) -> float:
    # C3 is 2:1 and C4 is wider; keep vertical framing consistent on both.
    return min(self._w, 2.0 * self._h)

  def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx3 device-frame points; returns (Nx2 screen points, depth)."""
    p = self._pose
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    dx = pts[:, 0] + p.back_m
    dy = pts[:, 1]
    dz = pts[:, 2] - (self._ground_z - p.height_m)
    cy_, sy_ = math.cos(self.yaw), math.sin(self.yaw)
    fx = dx * cy_ + dy * sy_
    fy = -dx * sy_ + dy * cy_
    th = math.radians(p.pitch_deg)
    ct, st = math.cos(th), math.sin(th)
    depth = fx * ct + dz * st
    down = -fx * st + dz * ct
    focal = p.focal_scale * self._focal_base()
    horizon = self._y0 + p.horizon_frac * self._h
    cy = horizon + focal * math.tan(th)
    safe = np.where(depth > 0.3, depth, np.nan)
    sx = self._x0 + self._w / 2 + focal * fy / safe
    sy = cy + focal * down / safe
    return np.stack((sx, sy), axis=1), depth

  def project_point(self, x: float, y: float, z: float):
    xy, depth = self.project(np.array([[x, y, z]]))
    if not depth[0] > 0.3:
      return None
    return float(xy[0, 0]), float(xy[0, 1])

  def scale_at(self, x: float, y: float, z: float) -> float:
    """Pixels per metre for a fronto-parallel object at this point."""
    _, depth = self.project(np.array([[x, y, z]]))
    if not depth[0] > 0.3:
      return 0.0
    return float(self._pose.focal_scale * self._focal_base() / depth[0])

  def ribbon(self, line: np.ndarray, half_width: float, z_offset: float, max_idx: int,
             max_distance: float | None = None, y_shift: float = 0.0, extend_back: bool = True) -> np.ndarray:
    """Left edge then reversed right edge, matching draw_polygon_solid input."""
    if line.shape[0] == 0:
      return np.empty((0, 2), dtype=np.float32)
    pts = line[:max_idx + 1].astype(np.float64)
    if max_distance is not None and 0 < max_idx < len(line) - 1:
      p0, p1 = line[max_idx], line[max_idx + 1]
      end = [max_distance, np.interp(max_distance, [p0[0], p1[0]], [p0[1], p1[1]]),
             np.interp(max_distance, [p0[0], p1[0]], [p0[2], p1[2]])]
      pts = np.vstack((pts, end))
    pts = pts[pts[:, 0] >= 0.0]
    if extend_back and len(pts) and pts[0, 0] > -0.2:
      # Extend the nearest model node slightly behind the camera reference so ribbons reach the bottom edge.
      back = pts[0].copy()
      back[0] = -0.2
      pts = np.vstack((back, pts))
    if len(pts) < 2:
      return np.empty((0, 2), dtype=np.float32)
    left = pts + np.array([0.0, y_shift - half_width, z_offset])
    right = pts + np.array([0.0, y_shift + half_width, z_offset])
    l_xy, l_d = self.project(left)
    r_xy, r_d = self.project(right)
    margin = 0.5 * self._w
    ok = ((l_d > 0.3) & (r_d > 0.3) & np.isfinite(l_xy).all(axis=1) & np.isfinite(r_xy).all(axis=1)
          & (l_xy[:, 1] >= self._y0 - margin) & (l_xy[:, 1] <= self._y0 + self._h + margin)
          & (np.abs(l_xy[:, 0] - (self._x0 + self._w / 2)) < 2.0 * self._w))
    l_xy, r_xy = l_xy[ok], r_xy[ok]
    if len(l_xy) < 2:
      return np.empty((0, 2), dtype=np.float32)
    return np.concatenate((l_xy, r_xy[::-1])).astype(np.float32)

  def car_box(self, x_rear: float, y: float, ground_z: float, max_width_frac: float = 0.24):
    """Screen centre, width and height for a car rear at the given point."""
    pt = self.project_point(x_rear, y, ground_z)
    if pt is None:
      return None
    scale = self.scale_at(x_rear, y, ground_z)
    vw = float(np.clip(CAR_WIDTH_M * scale, 0.025 * self._w, max_width_frac * self._w))
    vh = vw * (CAR_HEIGHT_M / CAR_WIDTH_M) * 1.06
    sx, ground_y = pt
    bottom = self._y0 + self._h * 0.94
    if ground_y > bottom:
      ground_y = bottom
    return sx, ground_y - vh * 0.5, vw, vh

  def ego_box(self):
    if self.zoom <= 0.02:
      return None
    return self.car_box(EGO_REAR_X_M, 0.0, self._ground_z, max_width_frac=0.30)
