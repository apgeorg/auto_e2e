"""Unit tests for the v0 V2X BEV rasteriser.

Pure numpy, fully synthetic — no datasets, no model, no extra deps. The headline
test (`test_neighbour_extends_ego_bev`) is the executable statement of the whole
feature: an object only a neighbour saw shows up in the ego BEV, flagged as
received-only.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_parsing.v2x import (
    CLASS,
    NUM_CHANNELS,
    N_OBSERVERS,
    OCCUPANCY,
    SOURCE,
    V2XMessage,
    V2XObject,
    rasterize_v2x,
)

_EGO_AT_ORIGIN = (0.0, 0.0, 0.0)
_RS = 256
_CENTER = _RS // 2


def _occupied_cells(grid: np.ndarray) -> int:
    return int(grid[OCCUPANCY].sum())


def test_output_shape_dtype_and_ranges():
    grid = rasterize_v2x(_EGO_AT_ORIGIN, [])
    assert grid.shape == (NUM_CHANNELS, _RS, _RS)
    assert grid.dtype == np.float32
    assert np.isfinite(grid).all()
    assert grid.min() >= 0.0 and grid.max() <= 1.0


def test_empty_messages_is_all_zero():
    grid = rasterize_v2x(_EGO_AT_ORIGIN, [])
    assert _occupied_cells(grid) == 0
    assert grid.sum() == 0.0


def test_ego_object_is_self_sourced():
    # Ego reports an object 10 m ahead -> occupied, source must stay 0.
    msg = V2XMessage(sender_pose=_EGO_AT_ORIGIN, is_ego=True,
                     objects=[V2XObject(x=10.0, y=0.0)])
    grid = rasterize_v2x(_EGO_AT_ORIGIN, [msg])
    assert _occupied_cells(grid) > 0
    assert grid[SOURCE].sum() == 0.0  # nothing is received-only


def test_neighbour_extends_ego_bev():
    """The core claim: a neighbour-only object appears in ego BEV as source=1."""
    # Ego sees nothing of its own; a neighbour reports an object 20 m ahead.
    neighbour = V2XMessage(sender_pose=_EGO_AT_ORIGIN, is_ego=False,
                           objects=[V2XObject(x=20.0, y=0.0)])
    grid = rasterize_v2x(_EGO_AT_ORIGIN, [neighbour])

    occupied = grid[OCCUPANCY] > 0
    assert occupied.any(), "neighbour object should occupy cells"
    # Every occupied cell here is beyond-LOS (received-only).
    assert np.array_equal(grid[SOURCE] > 0, occupied)
    # It lands ahead of ego -> upper half of the canvas (forward maps up).
    rows = np.where(occupied)[0]
    assert rows.mean() < _CENTER


def test_corroboration_clears_source_and_counts_observers():
    # Same object reported by ego AND a neighbour -> source 0, n_observers 2.
    obj = V2XObject(x=15.0, y=0.0)
    ego = V2XMessage(sender_pose=_EGO_AT_ORIGIN, is_ego=True, objects=[obj])
    nb = V2XMessage(sender_pose=_EGO_AT_ORIGIN, is_ego=False, objects=[obj])
    grid = rasterize_v2x(_EGO_AT_ORIGIN, [ego, nb], observer_norm=4.0)

    occupied = grid[OCCUPANCY] > 0
    assert grid[SOURCE].sum() == 0.0  # ego saw it too -> not received-only
    # Two agents corroborate -> 2 / observer_norm on the occupied cells.
    np.testing.assert_allclose(grid[N_OBSERVERS][occupied].max(), 2.0 / 4.0)


def test_out_of_range_object_is_dropped():
    # 200 m ahead is well outside the 60 m radius -> nothing painted.
    far = V2XMessage(sender_pose=_EGO_AT_ORIGIN, is_ego=False,
                     objects=[V2XObject(x=200.0, y=0.0)])
    grid = rasterize_v2x(_EGO_AT_ORIGIN, [far])
    assert _occupied_cells(grid) == 0


def test_neighbour_pose_is_applied():
    # Object at the neighbour's own origin; neighbour sits 20 m left of ego.
    # In the ego frame the object should land left of centre (column < centre).
    nb = V2XMessage(sender_pose=(0.0, 20.0, 0.0), is_ego=False,
                    objects=[V2XObject(x=0.0, y=0.0)])
    grid = rasterize_v2x(_EGO_AT_ORIGIN, [nb])
    occupied = grid[OCCUPANCY] > 0
    assert occupied.any()
    cols = np.where(occupied)[1]
    assert cols.mean() < _CENTER  # left of ego -> left on canvas


def test_class_channel_normalised():
    nb = V2XMessage(sender_pose=_EGO_AT_ORIGIN, is_ego=False,
                    objects=[V2XObject(x=10.0, y=0.0, cls=1)])  # pedestrian
    grid = rasterize_v2x(_EGO_AT_ORIGIN, [nb], class_norm=2.0)
    occupied = grid[OCCUPANCY] > 0
    np.testing.assert_allclose(grid[CLASS][occupied].max(), 1.0 / 2.0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
