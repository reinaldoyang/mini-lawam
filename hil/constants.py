"""Small shared constants kept free of robot and model dependencies."""

GRIPPER_OPEN = 0
GRIPPER_CLOSE = 1
GRIPPER_LABEL_MEANING = "0=open, 1=close"

ACTION_DIM = 7
ACTION_MEANING = "[dx, dy, dz, dRx, dRy, dRz, gripper_state]"
ACTION_SCHEMA = "mini_lawam_vr_hil_forward_command_delta_v1"
INTERVENTION_LABEL_SCHEMA = "vr_active_motion_or_gripper_edge_v2"

# The task locks roll and pitch; Stage 1 learns only the controllable arm axes.
ARM_CORRECTION_INDICES = (0, 1, 2, 5)
ARM_CORRECTION_MEANING = "[dx,dy,dz,dRz]"
