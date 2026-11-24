"""Stabilized radar + vision vehicle fusion and tracking for VR Mode.

Fuses `radarState` (leadOne, leadTwo, leadsLeft, leadsRight) with `liveTracks`
(displayPoints when supported, otherwise measured points) while aggressively
filtering stationary clutter (guardrails, road signs, poles) and damping radar
lateral/longitudinal jitter so vehicles glide smoothly in VR Mode.
"""
from dataclasses import dataclass
import math

import numpy as np

from openpilot.selfdrive.ui.radar_display import select_radar_display_points

MAX_VR_RANGE_M = 130.0
MIN_VR_RANGE_M = 0.8
MAX_VR_LATERAL_M = 11.5
MAX_VR_VEHICLES = 12

# Adjacent (non-lead) stationary rejection thresholds (m/s)
ADJ_ADMIT_MIN_VABS_MPS = 2.2   # ~8 km/h required to admit a new adjacent radar track
ADJ_KEEP_MIN_VABS_MPS = 1.3    # ~4.7 km/h hysteresis to retain an already-confirmed car
ADJ_CONFIRM_HITS = 2           # Require >= 2 updates to reject 1-frame radar ghosts
ADJ_CONFIRM_AGE_S = 0.10

# Lifecycle timing
TRACK_COAST_HOLD_S = 0.12      # Hold full opacity briefly across missed frames
TRACK_FADE_OUT_S = 0.35        # Smooth fade-out after dropout
TRACK_FADE_IN_S = 0.22         # Smooth fade-in on confirmation

# Spatial deduplication tolerances (m)
DEDUP_DX_M = 3.0
DEDUP_DY_M = 1.35
SCC_TWIN_RANGE_M = 1.5
SCC_TWIN_SPEED_MPS = 1.0

# Lateral anti-jitter limits
ADJ_MAX_LAT_SLEW_MPS = 1.85    # Cap apparent sideways jump rate for adjacent tracks


def _ema_alpha(dt: float, tau: float) -> float:
  if tau <= 0.0:
    return 1.0
  dt_eff = dt if dt > 0.0 else 0.05
  return float(1.0 - math.exp(-dt_eff / tau))


def _finite(val) -> float | None:
  try:
    f = float(val)
    return f if math.isfinite(f) else None
  except (TypeError, ValueError):
    return None


@dataclass(frozen=True)
class VrVehicle:
  track_key: str
  d_rel: float        # Forward distance (m)
  y_dev: float        # Lateral position in device frame (+right, -left) (m)
  v_rel: float        # Relative speed (m/s)
  v_abs: float        # Absolute speed (m/s)
  a_lead: float       # Longitudinal accel (m/s^2)
  tag: str            # 'L1', 'L2', or 'ADJ'
  opacity: float      # [0, 1] smooth fade-in / fade-out
  disp_d: float       # Hysteresis-stabilized distance for L1/L2 pill (m)
  braking: bool       # Brake light state


@dataclass
class _RawDetection:
  track_id: int | None
  d_rel: float
  y_dev: float        # +right, -left
  v_rel: float
  v_abs: float
  a_lead: float
  tag: str            # 'L1', 'L2', 'ADJ'
  priority: int       # Higher wins dedup: L1=3, L2=2, radarState adj=1, liveTracks=0
  on_path: bool = False


def _is_scc_twin(a: _RawDetection, b: _RawDetection) -> bool:
  return abs(a.d_rel - b.d_rel) < SCC_TWIN_RANGE_M and abs(a.v_rel - b.v_rel) < SCC_TWIN_SPEED_MPS


@dataclass
class _TrackState:
  key: str
  track_id: int | None
  tag: str
  d: float
  y: float
  v_rel: float
  v_abs: float
  a_lead: float
  disp_d: float
  first_t: float
  last_seen_t: float
  hits: int
  confirmed: bool
  opacity: float


def _path_y_at(path: np.ndarray | None, d: float) -> float:
  if path is None or path.shape[0] < 2:
    return 0.0
  return float(np.interp(d, path[:, 0], path[:, 1]))


def _edge_bounds_at(road_edges: list | None, road_edge_stds: np.ndarray | None,
                    d: float, max_std: float = 1.9) -> tuple[float | None, float | None]:
  if not road_edges or len(road_edges) < 2:
    return None, None
  left_y = right_y = None
  for idx, is_left in ((0, True), (1, False)):
    std = float(road_edge_stds[idx]) if road_edge_stds is not None and idx < len(road_edge_stds) else 1.0
    if std > max_std:
      continue
    raw = getattr(road_edges[idx], 'raw_points', road_edges[idx])
    if raw is None or getattr(raw, 'shape', (0,))[0] < 2:
      continue
    y_val = float(np.interp(d, raw[:, 0], raw[:, 1]))
    if is_left:
      left_y = y_val
    else:
      right_y = y_val
  return left_y, right_y


class VrTrackManager:
  """Maintains rock-solid, noise-damped vehicle tracks for VR Mode."""

  def __init__(self):
    self._tracks: dict[str, _TrackState] = {}
    self._anon_seq: int = 0

  def reset(self) -> None:
    self._tracks.clear()
    self._anon_seq = 0

  def _collect_detections(self, sm, v_ego: float, path: np.ndarray | None,
                          road_edges: list | None, road_edge_stds: np.ndarray | None,
                          started_frame: int = 0, now_t: float | None = None) -> list[_RawDetection]:
    raw: list[_RawDetection] = []
    radar_state = sm['radarState'] if getattr(sm, 'valid', {}).get('radarState', False) else None

    if radar_state is not None:
      for attr, tag, prio in (('leadOne', 'L1', 3), ('leadTwo', 'L2', 2)):
        lead = getattr(radar_state, attr, None)
        if lead is None or not bool(getattr(lead, 'status', False)):
          continue
        d_rel = _finite(getattr(lead, 'dRel', None))
        y_rel = _finite(getattr(lead, 'yRel', None))
        if d_rel is None or y_rel is None or not (MIN_VR_RANGE_M < d_rel <= MAX_VR_RANGE_M):
          continue
        v_rel = _finite(getattr(lead, 'vRel', 0.0)) or 0.0
        v_lead = _finite(getattr(lead, 'vLeadK', getattr(lead, 'vLead', None)))
        v_abs = v_lead if v_lead is not None else (v_ego + v_rel)
        a_lead = _finite(getattr(lead, 'aLead', getattr(lead, 'aLeadK', getattr(lead, 'aRel', 0.0)))) or 0.0
        tid_raw = getattr(lead, 'radarTrackId', -1)
        tid = int(tid_raw) if isinstance(tid_raw, (int, np.integer)) and int(tid_raw) >= 0 else None

        # Anchor on-path lead gently to predicted driving path on curves while preserving
        # genuine lateral offset (e.g. cut-in / cut-out) within a bounded band.
        y_dev_radar = -y_rel
        path_y = _path_y_at(path, d_rel)
        if path is not None and path.shape[0] >= 2:
          lat_err = float(np.clip(y_dev_radar - path_y, -0.85, 0.85))
          y_dev = path_y + 0.35 * lat_err
        else:
          y_dev = y_dev_radar

        is_radar = bool(getattr(lead, 'radar', False))
        d_path = _finite(getattr(lead, 'dPath', 0.0)) or 0.0
        v_lat = _finite(getattr(lead, 'vLat', 0.0)) or 0.0
        on_path = is_radar and abs(y_rel) < 1e-3 and abs(d_path) < 1e-3 and abs(v_lat) < 1e-3

        raw.append(_RawDetection(tid, d_rel, y_dev, v_rel, v_abs, a_lead, tag, prio, on_path=on_path))

      for attr in ('leadsLeft', 'leadsRight'):
        for lead in getattr(radar_state, attr, None) or []:
          if not bool(getattr(lead, 'status', False)):
            continue
          d_rel = _finite(getattr(lead, 'dRel', None))
          y_rel = _finite(getattr(lead, 'yRel', None))
          if d_rel is None or y_rel is None or not (1.5 < d_rel <= 110.0):
            continue
          y_dev = -y_rel
          if abs(y_dev) > MAX_VR_LATERAL_M:
            continue
          v_rel = _finite(getattr(lead, 'vRel', 0.0)) or 0.0
          v_lead = _finite(getattr(lead, 'vLeadK', getattr(lead, 'vLead', None)))
          v_abs = v_lead if v_lead is not None else (v_ego + v_rel)
          a_lead = _finite(getattr(lead, 'aLead', getattr(lead, 'aLeadK', 0.0))) or 0.0
          tid_raw = getattr(lead, 'radarTrackId', -1)
          tid = int(tid_raw) if isinstance(tid_raw, (int, np.integer)) and int(tid_raw) >= 0 else None
          is_radar = bool(getattr(lead, 'radar', False))
          d_path = _finite(getattr(lead, 'dPath', 0.0)) or 0.0
          v_lat = _finite(getattr(lead, 'vLat', 0.0)) or 0.0
          on_path = is_radar and abs(y_rel) < 1e-3 and abs(d_path) < 1e-3 and abs(v_lat) < 1e-3
          raw.append(_RawDetection(tid, d_rel, y_dev, v_rel, v_abs, a_lead, 'ADJ', 1, on_path=on_path))

    # Fuse liveTracks (prefer displayPoints when available, otherwise measured points)
    display_pts = select_radar_display_points(sm, show_radar_info=3, now=now_t, started_frame=started_frame)
    if display_pts is not None:
      for pt in display_pts:
        d_rel = float(pt.dRel)
        y_dev = -float(pt.yRel)
        if not (1.5 < d_rel <= 115.0) or abs(y_dev) > MAX_VR_LATERAL_M:
          continue
        v_abs = float(pt.vLeadK)
        v_rel = v_abs - v_ego
        raw.append(_RawDetection(None, d_rel, y_dev, v_rel, v_abs, 0.0, 'ADJ', 0, on_path=False))
    elif getattr(sm, 'valid', {}).get('liveTracks', False):
      lt = sm['liveTracks']
      for pt in getattr(lt, 'points', None) or []:
        if not bool(getattr(pt, 'measured', False)):
          continue
        d_rel = _finite(getattr(pt, 'dRel', None))
        y_rel = _finite(getattr(pt, 'yRel', None))
        if d_rel is None or y_rel is None or not (1.5 < d_rel <= 115.0):
          continue
        y_dev = -y_rel
        if abs(y_dev) > MAX_VR_LATERAL_M:
          continue
        v_rel = _finite(getattr(pt, 'vRel', None))
        v_lead = _finite(getattr(pt, 'vLead', None))
        if v_rel is None and v_lead is None:
          continue
        v_abs = v_lead if v_lead is not None else (v_ego + float(v_rel))
        v_rel_eff = float(v_rel) if v_rel is not None else (v_abs - v_ego)
        tid_raw = getattr(pt, 'trackId', -1)
        tid = int(tid_raw) if isinstance(tid_raw, (int, np.integer)) and int(tid_raw) >= 0 else None
        src = str(getattr(pt, 'radarSource', '') or '').lower()
        on_path = (src == 'scc')
        raw.append(_RawDetection(tid, d_rel, y_dev, v_rel_eff, v_abs, 0.0, 'ADJ', 0, on_path=on_path))

    # Merge SCC radar-only twins (on_path=True, zero lateral) with lateral radar tracks
    consumed: set[int] = set()
    for i, det in enumerate(raw):
      if i in consumed or not det.on_path or det.tag not in ('L1', 'L2'):
        continue
      best_j: int | None = None
      best_dd = 1e9
      for j, other in enumerate(raw):
        if j == i or j in consumed or other.on_path or not _is_scc_twin(other, det):
          continue
        dd = abs(other.d_rel - det.d_rel)
        if dd < best_dd:
          best_dd = dd
          best_j = j
      if best_j is not None:
        other = raw[best_j]
        if det.track_id is None and other.track_id is not None:
          det.track_id = other.track_id
        path_y = _path_y_at(path, det.d_rel)
        if path is not None and path.shape[0] >= 2:
          lat_err = float(np.clip(other.y_dev - path_y, -0.85, 0.85))
          det.y_dev = path_y + 0.35 * lat_err
        else:
          det.y_dev = other.y_dev
        det.on_path = False
        consumed.add(best_j)

    for i, det in enumerate(raw):
      if i in consumed or not det.on_path or det.tag != 'ADJ':
        continue
      has_twin = any(
        j != i and j not in consumed and not other.on_path and _is_scc_twin(other, det)
        for j, other in enumerate(raw)
      )
      if has_twin:
        consumed.add(i)

    if consumed:
      raw = [det for i, det in enumerate(raw) if i not in consumed]

    # Filter out off-road clutter (guardrails / signs outside road edges) for ADJ detections
    filtered: list[_RawDetection] = []
    for det in raw:
      if det.tag == 'ADJ':
        left_e, right_e = _edge_bounds_at(road_edges, road_edge_stds, det.d_rel)
        if left_e is not None and det.y_dev < left_e - 1.35:
          continue
        if right_e is not None and det.y_dev > right_e + 1.35:
          continue
        # Reject oncoming/reverse radar clutter near ego lane
        if det.v_abs < -2.0 and abs(det.y_dev - _path_y_at(path, det.d_rel)) < 2.6:
          continue
      filtered.append(det)

    # Deduplicate (higher priority wins: L1 > L2 > radarState ADJ > liveTracks)
    filtered.sort(key=lambda d: (-d.priority, d.d_rel))
    deduped: list[_RawDetection] = []
    for det in filtered:
      duplicate = False
      for kept in deduped:
        if det.track_id is not None and kept.track_id is not None and det.track_id == kept.track_id:
          duplicate = True
          break
        if abs(det.d_rel - kept.d_rel) < DEDUP_DX_M and abs(det.y_dev - kept.y_dev) < DEDUP_DY_M:
          duplicate = True
          break
        if det.tag == 'L2' and kept.tag == 'L1' and abs(det.d_rel - kept.d_rel) < 3.5:
          duplicate = True
          break
      if not duplicate:
        deduped.append(det)
    return deduped

  def _match_track(self, det: _RawDetection, claimed: set[str]) -> _TrackState | None:
    if det.tag in ('L1', 'L2'):
      tr = self._tracks.get(det.tag)
      if tr is not None and tr.key not in claimed:
        return tr
      return None

    if det.track_id is not None:
      key = f"T{det.track_id}"
      tr = self._tracks.get(key)
      if tr is not None and tr.key not in claimed:
        return tr

    # Nearest-neighbor association for tracks without stable ID
    best: _TrackState | None = None
    best_cost = 1e9
    for key, tr in self._tracks.items():
      if key in claimed or tr.tag != 'ADJ':
        continue
      dd = abs(det.d_rel - tr.d)
      dy = abs(det.y_dev - tr.y)
      if dd < 4.5 and dy < 1.8:
        cost = dd + 2.5 * dy
        if cost < best_cost:
          best_cost = cost
          best = tr
    return best

  def update(self, sm, v_ego: float, path: np.ndarray | None,
             road_edges: list | None, road_edge_stds: np.ndarray | None,
             dt: float, now_t: float, started_frame: int = 0) -> list[VrVehicle]:
    detections = self._collect_detections(sm, v_ego, path, road_edges, road_edge_stds,
                                          started_frame=started_frame, now_t=now_t)
    stationary_ego = v_ego < 0.35
    claimed: set[str] = set()

    for det in detections:
      tr = self._match_track(det, claimed)

      # Strict stationary rejection for adjacent tracks (prevents guardrail/sign false cars)
      if det.tag == 'ADJ':
        min_vabs = ADJ_KEEP_MIN_VABS_MPS if (tr is not None and tr.confirmed) else ADJ_ADMIT_MIN_VABS_MPS
        if det.v_abs < min_vabs:
          continue

      if tr is None:
        if det.tag in ('L1', 'L2'):
          key = det.tag
        elif det.track_id is not None:
          key = f"T{det.track_id}"
        else:
          self._anon_seq += 1
          key = f"A{self._anon_seq}"
        confirmed = det.tag in ('L1', 'L2')
        tr = _TrackState(
          key=key,
          track_id=det.track_id,
          tag=det.tag,
          d=det.d_rel,
          y=det.y_dev,
          v_rel=det.v_rel,
          v_abs=det.v_abs,
          a_lead=det.a_lead,
          disp_d=round(det.d_rel, 1),
          first_t=now_t,
          last_seen_t=now_t,
          hits=1,
          confirmed=confirmed,
          opacity=1.0 if confirmed else 0.0,
        )
        self._tracks[key] = tr
        claimed.add(key)
        continue

      # Reset jump if lead switched to a completely different car
      if abs(det.d_rel - tr.d) > 8.0 or abs(det.y_dev - tr.y) > 3.2:
        tr.d = det.d_rel
        tr.y = det.y_dev
        tr.v_rel = det.v_rel
        tr.v_abs = det.v_abs
        tr.a_lead = det.a_lead
        tr.disp_d = round(det.d_rel, 1)
      else:
        if stationary_ego and abs(det.v_rel) < 0.5:
          if abs(det.d_rel - tr.d) >= 0.50:
            tr.d += (det.d_rel - tr.d) * _ema_alpha(dt, 0.75)
          if abs(det.y_dev - tr.y) >= 0.25:
            tr.y += (det.y_dev - tr.y) * _ema_alpha(dt, 0.85)
        else:
          tau_d = 0.20 if det.tag in ('L1', 'L2') else 0.28
          tau_y = 0.24 if det.tag in ('L1', 'L2') else 0.40
          tr.d += (det.d_rel - tr.d) * _ema_alpha(dt, tau_d)
          dy = det.y_dev - tr.y
          deadband = 0.06 if det.tag in ('L1', 'L2') else 0.12
          if abs(dy) > deadband:
            step_y = dy * _ema_alpha(dt, tau_y)
            if det.tag == 'ADJ' and dt > 0.0:
              max_step = ADJ_MAX_LAT_SLEW_MPS * dt
              step_y = float(np.clip(step_y, -max_step, max_step))
            tr.y += step_y
        tr.v_rel += (det.v_rel - tr.v_rel) * _ema_alpha(dt, 0.30)
        tr.v_abs += (det.v_abs - tr.v_abs) * _ema_alpha(dt, 0.30)
        tr.a_lead += (det.a_lead - tr.a_lead) * _ema_alpha(dt, 0.28)

      if abs(tr.d - tr.disp_d) >= 0.35:
        tr.disp_d = round(tr.d, 1)

      tr.hits += 1
      tr.last_seen_t = now_t
      if not tr.confirmed and (tr.hits >= ADJ_CONFIRM_HITS or (now_t - tr.first_t) >= ADJ_CONFIRM_AGE_S):
        tr.confirmed = True
      if tr.confirmed:
        step_op = (dt / TRACK_FADE_IN_S) if dt > 0.0 else 1.0
        tr.opacity = min(1.0, tr.opacity + step_op)
      claimed.add(tr.key)

    # Coast and smoothly fade out temporarily missing tracks; prune expired ones
    to_delete: list[str] = []
    for key, tr in self._tracks.items():
      if key in claimed:
        continue
      age = now_t - tr.last_seen_t
      if not tr.confirmed:
        if age > 0.20:
          to_delete.append(key)
        continue
      if dt > 0.0 and not stationary_ego:
        tr.d = max(MIN_VR_RANGE_M, tr.d + tr.v_rel * dt)
      if age > TRACK_COAST_HOLD_S:
        fade_frac = (age - TRACK_COAST_HOLD_S) / TRACK_FADE_OUT_S
        tr.opacity = max(0.0, 1.0 - fade_frac)
      if tr.opacity <= 0.01 or age > (TRACK_COAST_HOLD_S + TRACK_FADE_OUT_S + 0.1):
        to_delete.append(key)
    for key in to_delete:
      del self._tracks[key]

    # Build output list (keep L1/L2 always, cap ADJ to MAX_VR_VEHICLES total, sorted far -> near)
    confirmed_tracks = [tr for tr in self._tracks.values() if tr.confirmed and tr.opacity > 0.02]
    leads = [tr for tr in confirmed_tracks if tr.tag in ('L1', 'L2')]
    adjs = sorted((tr for tr in confirmed_tracks if tr.tag == 'ADJ'), key=lambda t: t.d)
    selected = leads + adjs[:max(0, MAX_VR_VEHICLES - len(leads))]
    selected.sort(key=lambda t: -t.d)

    out: list[VrVehicle] = []
    for tr in selected:
      braking = tr.tag != 'ADJ' and (
        tr.a_lead < -0.45 or (tr.a_lead < -0.25 and tr.v_rel < -0.3) or tr.v_rel < -1.2
      )
      out.append(VrVehicle(
        track_key=tr.key,
        d_rel=tr.d,
        y_dev=tr.y,
        v_rel=tr.v_rel,
        v_abs=tr.v_abs,
        a_lead=tr.a_lead,
        tag=tr.tag,
        opacity=tr.opacity,
        disp_d=tr.disp_d,
        braking=braking,
      ))
    return out
