from __future__ import annotations

import math

import numpy as np

from hil.policy import TemporalEnsembler


def test_temporal_ensemble_matches_rollout_age_weighting() -> None:
    ensemble = TemporalEnsembler(
        enabled=True,
        decay=math.log(2.0),
        gripper_threshold=0.0,
        open_lead_steps=0,
    )
    first = np.asarray(
        [[1.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    second = np.asarray(
        [[2.0, 0.0, 0.0, 1.0], [20.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    np.testing.assert_allclose(ensemble.select(first), first[0], atol=1e-7)
    selected = ensemble.select(second)

    # At absolute step 1, the old prediction has weight 0.5 and the new one 1.0.
    np.testing.assert_allclose(selected[:3], [14.0 / 3.0, 0.0, 0.0], atol=1e-6)
    assert selected[-1] == 1.0


def test_gripper_uses_newest_chunk_and_latches_release() -> None:
    ensemble = TemporalEnsembler(
        enabled=False,
        decay=0.2,
        gripper_threshold=0.0,
        open_lead_steps=1,
    )
    close = np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    opening_ahead = np.asarray(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]],
        dtype=np.float32,
    )

    assert ensemble.select(close)[-1] == 1.0
    assert ensemble.select(opening_ahead)[-1] == -1.0
    assert ensemble.select(close)[-1] == -1.0

