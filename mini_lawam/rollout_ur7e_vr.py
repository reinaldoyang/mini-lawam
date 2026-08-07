#!/usr/bin/env python3
"""UR7e rollout for VR-collected command-delta checkpoints.

Unlike the keyboard-compatible rollout, each predicted XYZ delta advances a
persistent commanded target. The target is initialized from the measured TCP
at rollout start and remains bounded by rollout_ur7e's workspace and
``--max-reach`` safety clamps relative to the current measured TCP.

Run from the LaWAM repository root:

    python -m mini_lawam.rollout_ur7e_vr --ckpt <VR_CHECKPOINT> ...
"""

from mini_lawam.rollout_ur7e import main


if __name__ == "__main__":
    main(joystick_anchor_mode="commanded", description=__doc__)
