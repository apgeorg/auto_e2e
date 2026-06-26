"""Rasterize received V2X messages into an ego-centric BEV tensor (v0).

This is the first step of the V2X *cooperative-perception* feature: neighbours
(other vehicles / infrastructure) broadcast the objects **they** perceive, and
the ego vehicle stitches them into its own bird's-eye-view so it can "see" past
occlusions and beyond camera range.

Frame contract — identical to the navigation-map raster (``kit_scenes/map.py``)
so the V2X grid lines up pixel-for-pixel with ``map_input``:

* ego-centred, ego heading points **up**: forward (+X_ego) -> row up,
  left (+Y_ego) -> column left (see ``_ego_to_px``, mirrors ``map._to_px``);
* a square window of ``2 * radius_meters`` (default 120 m) at ``canvas_size`` px,
  so ``scale = canvas_size / (2 * radius_meters)``.

Channel layout (v0 — kinematics deliberately omitted; poses assumed exact):

===  ============  ==================================================
ch   name          meaning
===  ============  ==================================================
0    occupancy     object footprints (ego + neighbours merged), 0/1
1    class          class scalar / ``class_norm`` (veh=0, ped=1, cyc=2)
2    source         0 = ego also saw it, 1 = **received-only** (beyond-LOS)
3    n_observers    corroborating agents / ``observer_norm``, clipped 0..1
===  ============  ==================================================

Channel 2 is the whole point: a cell that is occupied with ``source == 1`` is
something the ego's own cameras never saw — the extended part of the BEV.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Channel indices (v0).
OCCUPANCY = 0
CLASS = 1
SOURCE = 2
N_OBSERVERS = 3
NUM_CHANNELS = 4

DEFAULT_CANVAS_SIZE = 256
DEFAULT_RADIUS_M = 60.0


@dataclass
class V2XObject:
    """A single perceived object, in the *reporting agent's* local frame.

    ``x`` is forward, ``y`` is left (metres). ``cls`` is a small integer class
    id (0 = vehicle, 1 = pedestrian, 2 = cyclist). ``length``/``width`` are the
    footprint extents in metres (axis-aligned in the ego frame for v0).
    """

    x: float
    y: float
    cls: int = 0
    length: float = 4.5
    width: float = 2.0


@dataclass
class V2XMessage:
    """One received message: who sent it, where they are, and what they saw."""

    sender_pose: tuple[float, float, float]  # (x, y, yaw) in the common frame
    objects: list[V2XObject] = field(default_factory=list)
    is_ego: bool = False  # True for the ego's own detections


def _sender_to_world(local_xy: np.ndarray, sender_pose: tuple[float, float, float]) -> np.ndarray:
    """Rotate+translate object coords from a sender's frame into the common frame."""
    sx, sy, syaw = sender_pose
    c, s = np.cos(syaw), np.sin(syaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local_xy @ rot.T + np.array([sx, sy], dtype=np.float64)


def _world_to_ego(world_xy: np.ndarray, ego_pose: tuple[float, float, float]) -> np.ndarray:
    """Common frame -> ego frame (X forward, Y left)."""
    ex, ey, eyaw = ego_pose
    c, s = np.cos(-eyaw), np.sin(-eyaw)
    rel = world_xy - np.array([ex, ey], dtype=np.float64)
    x_rot = c * rel[:, 0] - s * rel[:, 1]
    y_rot = s * rel[:, 0] + c * rel[:, 1]
    return np.stack([x_rot, y_rot], axis=1)


def _ego_to_px(ego_xy: np.ndarray, scale: float, rs: int) -> np.ndarray:
    """Ego-frame metres -> (row, col) pixels; mirrors ``map._to_px``."""
    cx = rs / 2.0
    col = cx - ego_xy[:, 1] * scale
    row = cx - ego_xy[:, 0] * scale
    return np.stack([row, col], axis=1)


def _stamp(mask: np.ndarray, row: float, col: float,
           half_len_px: float, half_wid_px: float) -> bool:
    """Paint an axis-aligned footprint; returns True if anything was painted.

    The object centre must fall inside the canvas, so out-of-range reports are
    dropped rather than smeared against the border.
    """
    rs = mask.shape[0]
    if not (0.0 <= row < rs and 0.0 <= col < rs):
        return False
    half_len_px = max(half_len_px, 0.5)
    half_wid_px = max(half_wid_px, 0.5)
    r0 = max(0, int(round(row - half_len_px)))
    r1 = min(rs, int(round(row + half_len_px)) + 1)
    c0 = max(0, int(round(col - half_wid_px)) )
    c1 = min(rs, int(round(col + half_wid_px)) + 1)
    if r0 >= r1 or c0 >= c1:
        return False
    mask[r0:r1, c0:c1] = True
    return True


def rasterize_v2x(
    ego_pose: tuple[float, float, float],
    messages: list[V2XMessage],
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    radius_meters: float = DEFAULT_RADIUS_M,
    class_norm: float = 2.0,
    observer_norm: float = 4.0,
) -> np.ndarray:
    """Render received V2X messages into a ``(4, canvas_size, canvas_size)`` BEV.

    Args:
        ego_pose: ego ``(x, y, yaw)`` in the common frame — the grid centre.
        messages: received messages (include the ego's own detections as a
            message with ``is_ego=True`` so the ``source`` channel is correct).
        canvas_size: output side length in pixels.
        radius_meters: half-width of the window (metres); 60 -> a 120 m square.
        class_norm: divisor that maps the class id into ~[0, 1].
        observer_norm: divisor (then clip) for the corroboration count.

    Returns:
        ``float32`` array of shape ``(NUM_CHANNELS, canvas_size, canvas_size)``.
    """
    rs = canvas_size
    scale = rs / (2.0 * radius_meters)

    occ = np.zeros((rs, rs), dtype=bool)
    cls = np.zeros((rs, rs), dtype=np.float32)
    ego_seen = np.zeros((rs, rs), dtype=bool)
    recv_seen = np.zeros((rs, rs), dtype=bool)
    observers = np.zeros((rs, rs), dtype=np.float32)

    for msg in messages:
        msg_mask = np.zeros((rs, rs), dtype=bool)
        for obj in msg.objects:
            local = np.array([[obj.x, obj.y]], dtype=np.float64)
            world = _sender_to_world(local, msg.sender_pose)
            ego_xy = _world_to_ego(world, ego_pose)
            px = _ego_to_px(ego_xy, scale, rs)[0]
            footprint = np.zeros((rs, rs), dtype=bool)
            painted = _stamp(
                footprint, px[0], px[1],
                half_len_px=0.5 * obj.length * scale,
                half_wid_px=0.5 * obj.width * scale,
            )
            if not painted:
                continue
            occ |= footprint
            cls[footprint] = obj.cls / class_norm
            msg_mask |= footprint
            if msg.is_ego:
                ego_seen |= footprint
            else:
                recv_seen |= footprint
        observers += msg_mask.astype(np.float32)

    out = np.zeros((NUM_CHANNELS, rs, rs), dtype=np.float32)
    out[OCCUPANCY] = occ.astype(np.float32)
    out[CLASS] = cls
    out[SOURCE] = (recv_seen & ~ego_seen).astype(np.float32)
    out[N_OBSERVERS] = np.clip(observers / observer_norm, 0.0, 1.0)
    return out
