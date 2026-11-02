"""Bounded longitudinal response to a strong early corner-radar CUT-IN risk."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


CUTIN_PREDECEL_MIN_SCORE = 0.15
CUTIN_PREDECEL_MIN_TTC_S = 2.5
CUTIN_PREDECEL_MAX_TTC_S = 8.0
CUTIN_PREDECEL_MIN_ACCEL_LIMIT = -0.65
CUTIN_PREDECEL_MAX_ACCEL_LIMIT = -0.25
CUTIN_PREDECEL_MAX_JERK_STEP = 0.15
NORMAL_MAX_ACCEL_STEP = 0.05

LEAD_PREDECEL_MIN_TTC_S = 2.5
LEAD_PREDECEL_MAX_TTC_S = 8.0
LEAD_PREDECEL_MIN_ACCEL_LIMIT = -0.50
LEAD_PREDECEL_MAX_ACCEL_LIMIT = -0.15



def get_cutin_predecel_accel_limit(radar_state: Any) -> float | None:
  """Return a bounded negative acceleration ceiling for strong early risk."""
  lead = getattr(radar_state, "leadCutInRisk", None)
  if (
    lead is None
    or not bool(getattr(lead, "status", False))
    or not bool(getattr(lead, "radar", False))
    or float(getattr(lead, "score", 0.0)) < CUTIN_PREDECEL_MIN_SCORE
  ):
    return None
  d_rel = float(getattr(lead, "dRel", 0.0))
  v_rel = float(getattr(lead, "vRel", 0.0))
  if not (
    math.isfinite(d_rel)
    and math.isfinite(v_rel)
    and d_rel > 0.0
    and v_rel < 0.0
  ):
    return None
  ttc_s = min(
    CUTIN_PREDECEL_MAX_TTC_S,
    max(CUTIN_PREDECEL_MIN_TTC_S, d_rel / max(-v_rel, 0.1)),
  )
  fraction = (
    (ttc_s - CUTIN_PREDECEL_MIN_TTC_S)
    / (CUTIN_PREDECEL_MAX_TTC_S - CUTIN_PREDECEL_MIN_TTC_S)
  )
  return (
    CUTIN_PREDECEL_MIN_ACCEL_LIMIT
    + fraction * (
      CUTIN_PREDECEL_MAX_ACCEL_LIMIT
      - CUTIN_PREDECEL_MIN_ACCEL_LIMIT
    )
  )


def get_lead_predecel_accel_limit(radar_state: Any, v_ego: float) -> float | None:
  """Return a bounded negative acceleration ceiling when following a lead car."""
  lead = getattr(radar_state, "leadOne", None)
  if (
    lead is None
    or not bool(getattr(lead, "status", False))
    or not bool(getattr(lead, "radar", False))
    or float(getattr(lead, "radarTrackId", -1)) < 0
  ):
    return None

  d_rel = float(getattr(lead, "dRel", 0.0))
  v_rel = float(getattr(lead, "vRel", 0.0))
  a_lead = float(getattr(lead, "aLeadK", 0.0))
  v_lead = float(getattr(lead, "vLead", v_ego + v_rel))

  # Active when driving, lead is at a reasonable distance and closing
  if not (
    math.isfinite(d_rel)
    and math.isfinite(v_rel)
    and math.isfinite(a_lead)
    and math.isfinite(v_ego)
    and v_ego > 1.0
    and d_rel > 3.0
    and v_rel < -0.3
  ):
    return None

  # Case 1: Stationary / stopped lead ahead while traveling at speed
  # Initiates smooth, progressive early deceleration well in advance with +1.0m buffer
  is_stationary_lead = (
    v_lead < 1.5
    and v_rel < -0.8
    and d_rel > 6.0
    and (v_ego > 5.0 or d_rel < 20.0)
  )

  # Case 2: Lead actively hard braking or urgent closing rate
  ttc_s = d_rel / max(-v_rel, 0.1)
  if v_ego < 16.0:
    is_hard_braking_lead = (
      (a_lead <= -1.5 and v_rel < -1.5 and ttc_s < 6.0)
      or (v_rel < -3.5 and ttc_s < 4.5)
    )
  else:
    is_hard_braking_lead = (
      (a_lead < -1.5 and v_rel < -1.5 and ttc_s < 6.0)
      or (v_rel < -4.0 and ttc_s < 4.5)
    )

  # Case 3: Lead decelerating towards a stop or low-speed stopping approach ahead
  # Initiates early progressive braking so gap does not collapse into collision risk
  is_stopping_lead = (
    not is_hard_braking_lead
    and v_lead < 7.0
    and (a_lead < -0.5 or (v_lead < 3.5 and v_rel < -0.4))
    and v_rel < -0.4
    and 3.5 < d_rel < 25.0
  )

  # Case 4: Closing in traffic at noticeable rate
  is_closing_follow = (
    not is_stationary_lead
    and not is_hard_braking_lead
    and not is_stopping_lead
    and v_ego > 5.0
    and d_rel >= 10.0
    and ((a_lead < -0.4 and v_rel < -2.0 and ttc_s < 7.0) or (v_rel < -4.0 and ttc_s < 6.0))
  )

  if not (is_stationary_lead or is_hard_braking_lead or is_stopping_lead or is_closing_follow):
    return None

  if is_stationary_lead:
    # Smooth, comfortable early deceleration to bring speed down early for stationary vehicle
    # Target stopping distance cushion 7.0m (+1.0m buffer)
    stop_target_dist = 7.0
    a_stopping_req = -(v_ego ** 2) / (2.0 * max(d_rel - stop_target_dist, 3.0))
    # Cap between -1.5 m/s^2 (comfortable progressive decel) and -0.4 m/s^2
    return max(-1.5, min(-0.4, a_stopping_req))
  elif is_hard_braking_lead:
    # Hard braking lead: match lead decel + closing rate, bounded to speed-appropriate floor
    a_rel_req = a_lead - (v_rel ** 2) / (2.0 * max(d_rel - 6.0, 4.0))
    min_floor = float(np.interp(v_ego, [4.0, 10.0, 16.0], [-1.2, -2.0, -2.5]))
    return max(min_floor, min(-0.5, a_rel_req))
  elif is_stopping_lead:
    # Low-speed decelerating lead: smoothly ramp deceleration to stop at +1.0m buffer
    stop_target_dist = 7.0
    a_stop_req = a_lead - (max(0.0, v_ego - v_lead) ** 2) / (2.0 * max(d_rel - stop_target_dist, 2.5))
    min_floor = float(np.interp(v_ego, [2.0, 6.7, 12.0], [-0.85, -0.95, -2.0]))
    return max(min_floor, min(-0.35, a_stop_req))
  else:
    # Gentle closing follow: smooth negative ceiling between -0.50 and -0.15 m/s^2
    clamped_ttc = min(
      LEAD_PREDECEL_MAX_TTC_S,
      max(LEAD_PREDECEL_MIN_TTC_S, ttc_s),
    )
    fraction = (
      (clamped_ttc - LEAD_PREDECEL_MIN_TTC_S)
      / (LEAD_PREDECEL_MAX_TTC_S - LEAD_PREDECEL_MIN_TTC_S)
    )
    return (
      LEAD_PREDECEL_MIN_ACCEL_LIMIT
      + fraction * (
        LEAD_PREDECEL_MAX_ACCEL_LIMIT
        - LEAD_PREDECEL_MIN_ACCEL_LIMIT
      )
    )


def apply_cutin_predecel_accel_limit(
  maximum_accel: float,
  a_desired: float,
  predecel_limit: float | None,
) -> float:
  """Apply the risk ceiling with a bounded per-plan acceleration step."""
  requested = (
    float(maximum_accel)
    if predecel_limit is None
    else min(float(maximum_accel), float(predecel_limit))
  )
  maximum_step = (
    0.08  # smooth progressive ramp (1.6 m/s^3) for urgent decel without harsh jerk
    if (predecel_limit is not None and predecel_limit < -1.0)
    else (
      NORMAL_MAX_ACCEL_STEP
      if predecel_limit is None
      else CUTIN_PREDECEL_MAX_JERK_STEP
    )
  )
  return max(requested, float(a_desired) - maximum_step)
