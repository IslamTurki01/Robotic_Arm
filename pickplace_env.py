"""Gymnasium env: UR5e arm + Robotiq 2F-85 picks up a cube and drops it in a box.

Scene defined in robot_pickplace.xml (UR5e + 2F-85 from Menagerie + cube + tray).
Action:
    joint mode      (7-D): 6 delta joint targets + 1 delta gripper command.
    cartesian mode  (4-D): TCP xyz delta, converted through IK, + gripper delta.
Obs   (29-D): arm q/qv (12) + gripper drive q/qv (2) + tip xyz (3)
              + cube xyz (3) + box xyz (3) + (cube - tip) (3) + (box - cube) (3).
Success: cube center is inside the box footprint and below the box rim.
"""

from __future__ import annotations

import os
from typing import Optional

import mujoco
import mujoco.viewer as mj_viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces

XML_PATH = os.path.join(os.path.dirname(__file__), "robot_pickplace.xml")

# Box geometry (must match robot_pickplace.xml). Inner half-extent of the
# tray footprint and rim height — used for the in-box success check.
BOX_INNER_HALF  = 0.06
BOX_RIM_HEIGHT  = 0.05
BOX_FLOOR_TOP_Z = 0.005           # cube rests on this surface when settled

# Cube geometry
CUBE_HALF = 0.015                 # 3 cm cube → half-size 0.015

# Workspace ranges (xy on the table, in robot frame)
CUBE_XY_LOW  = np.array([0.40, -0.22], dtype=np.float64)
CUBE_XY_HIGH = np.array([0.55, -0.05], dtype=np.float64)
BOX_XY_LOW   = np.array([0.40,  0.05], dtype=np.float64)
BOX_XY_HIGH  = np.array([0.55,  0.22], dtype=np.float64)

# Actuator index layout (see robot_pickplace.xml <actuator> section)
ARM_ACT_SLICE = slice(0, 6)
GRIP_ACT_IDX  = 6
GRIP_CTRL_MIN = 0.0
GRIP_CTRL_MAX = 255.0

# Scaling for the user-facing 7-D action space
ARM_DELTA_LIMIT  = 0.08
GRIP_DELTA_LIMIT = 20.0
TCP_DELTA_LIMIT  = 0.035

# Cartesian action mode converts TCP deltas into joint deltas via damped IK.
CART_IK_POS_GAIN = 1.0
CART_IK_ROT_GAIN = 0.35
CART_IK_DAMPING  = 0.05

# Lift threshold: cube must clear this height for "lifted" bonus
LIFT_HEIGHT = 0.04

# ----------------------------------------------------------------------------
# Stage-gated reward — absolute axis-weighted distance + sparse terminal bonus.
# Design rationale:
#   * Task decomposition: only ONE shaping term active per step (Park & Won
#     2023, Biomimetics 8(2):240 → 93.2% success on pick-and-place).
#   * Each stage's shaping is bounded above by 0, so the agent never gets
#     net-positive per-step reward by loitering — it must terminate to win.
#   * Dense magnitudes are tuned so a successful episode totals ~+60–80 vs
#     -300+ for a stalled one. The +100 terminal cleanly dominates shaping.
#   * No PBRS: in our prior run PBRS gave Bellman-correct shaping but a too-
#     weak exploration gradient (~0.013/step bias vs −0.10/step living cost).
#     Absolute distance gives ~50× stronger signal toward the cube.
# ----------------------------------------------------------------------------
W_REACH_XY     = 1.0     # stage-0 xy weight (axis-priority: align before descend)
W_REACH_Z      = 0.5     # stage-0 z weight
W_LIFT         = 5.0     # stage-1 weight on lift gap
W_PLACE_XY     = 1.5     # stage-2 cube↔box xy weight
LIFT_TARGET_Z  = 0.10    # stage-1 saturates once cube z reaches this
GRASP_BONUS    = 5.0     # one-shot on first reaching stage 1
LIFT_BONUS     = 10.0    # one-shot on first reaching stage 2
SUCCESS_BONUS  = 100.0   # one-shot terminal: cube settled in box
DROP_PENALTY   = 10.0    # each time grasp is lost outside the box
PUSH_PENALTY   = 2.0     # arm collides with cube without gripping
REACH_BONUS     = 1.0    # one-shot when TCP first reaches the grasp window
LEFT_PAD_BONUS  = 1.0    # one-shot for first left-pad contact
RIGHT_PAD_BONUS = 3.0    # one-shot, larger because prior runs missed it
BOTH_PAD_BONUS  = 3.0    # one-shot for first simultaneous two-pad contact

# Grasp-discovery shaping. These are costs, not per-step positive rewards.
GRASP_WINDOW_XY = 0.035
GRASP_WINDOW_Z  = 0.035
GRIPPER_Q_MAX   = 0.80
HOVER_Z         = 0.10
PRE_GRASP_Z_OFFSET = 0.035


def mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to axis-angle vector."""
    cos_t = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_t))
    if theta < 1e-6:
        return np.zeros(3)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))
    return axis * theta


class PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 400,
        curriculum_level: int = 2,
        action_mode: str = "joint",
        near_cube_reset_prob: float = 0.0,
        pregrasp_reset_prob: float = 0.0,
    ):
        super().__init__()

        if action_mode not in ("joint", "cartesian"):
            raise ValueError("action_mode must be 'joint' or 'cartesian'")

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)

        self.max_steps  = max_steps
        self._step_count = 0
        self.render_mode = render_mode
        self._viewer = None
        self.curriculum_level = int(curriculum_level)
        self.action_mode = action_mode
        self.near_cube_reset_prob = float(np.clip(near_cube_reset_prob, 0.0, 1.0))
        self.pregrasp_reset_prob = float(np.clip(pregrasp_reset_prob, 0.0, 1.0))

        # --- Lookup IDs / qpos addresses we'll use every step ----------------
        self.pinch_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch"
        )

        # UR5e joint qpos addresses (first 6)
        arm_joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ]
        self.arm_qpos_adr = np.array([
            self.model.joint(n).qposadr[0] for n in arm_joint_names
        ])
        self.arm_dof_adr = np.array([
            self.model.joint(n).dofadr[0] for n in arm_joint_names
        ])

        # Robotiq driver qpos
        self.right_driver_qpos_adr = self.model.joint("right_driver_joint").qposadr[0]
        self.right_driver_dof_adr  = self.model.joint("right_driver_joint").dofadr[0]

        # Cube free joint (qpos: px py pz qw qx qy qz; qvel: 6 dofs)
        self.cube_qpos_adr = self.model.joint("cube_free").qposadr[0]
        self.cube_dof_adr  = self.model.joint("cube_free").dofadr[0]
        self.cube_bid      = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )

        # Mocap box body
        self.box_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_box"
        )
        self.box_mocap_id = int(self.model.body_mocapid[self.box_bid])

        # Geom ids for contact-based grasp detection
        gid = lambda n: mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, n
        )
        self.cube_geom_id   = gid("cube_geom")
        self.left_pad_ids   = {gid("left_pad1"),  gid("left_pad2")}
        self.right_pad_ids  = {gid("right_pad1"), gid("right_pad2")}

        # Body ids for orientation / bracket-the-cube rewards
        bid = lambda n: mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, n
        )
        self.gripper_base_bid = bid("gripper_base")
        self.left_pad_bid     = bid("left_pad")
        self.right_pad_bid    = bid("right_pad")

        self._home_tool_R = self._compute_home_tool_R()

        # --- Spaces ----------------------------------------------------------
        if self.action_mode == "joint":
            action_low = [-ARM_DELTA_LIMIT] * 6 + [-GRIP_DELTA_LIMIT]
            action_high = [ARM_DELTA_LIMIT] * 6 + [GRIP_DELTA_LIMIT]
        else:
            action_low = [-TCP_DELTA_LIMIT] * 3 + [-GRIP_DELTA_LIMIT]
            action_high = [TCP_DELTA_LIMIT] * 3 + [GRIP_DELTA_LIMIT]
        self.action_space = spaces.Box(
            low=np.array(action_low, dtype=np.float32),
            high=np.array(action_high, dtype=np.float32),
        )
        obs_dim = 6 + 6 + 1 + 1 + 3 + 3 + 3 + 3 + 3   # = 29
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.ctrl_low  = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        # Stage-gated reward state (filled in by reset())
        self._prev_stage = 0
        self._max_stage  = 0
        self._prev_grasp = False
        self._reach_seen = False
        self._left_touch_seen = False
        self._right_touch_seen = False
        self._both_touch_seen = False
        self._reset_start_mode = "home"

    # =====================================================================
    def set_curriculum_level(self, level: int):
        self.curriculum_level = int(level)

    def set_reset_curriculum(
        self,
        near_cube_reset_prob: float,
        pregrasp_reset_prob: float,
    ):
        self.near_cube_reset_prob = float(np.clip(near_cube_reset_prob, 0.0, 1.0))
        self.pregrasp_reset_prob = float(np.clip(pregrasp_reset_prob, 0.0, 1.0))

    def _compute_home_tool_R(self) -> np.ndarray:
        tmp_data = mujoco.MjData(self.model)
        ready = np.array([0.0, -1.6, 1.6, -1.6, -1.57, 0.0])
        tmp_data.qpos[self.arm_qpos_adr] = ready
        mujoco.mj_forward(self.model, tmp_data)
        return tmp_data.site_xmat[self.pinch_site_id].reshape(3, 3).copy()

    # =====================================================================
    def _cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_bid].copy()

    def _box_pos(self) -> np.ndarray:
        return self.data.mocap_pos[self.box_mocap_id].copy()

    def _get_obs(self) -> np.ndarray:
        arm_q  = self.data.qpos[self.arm_qpos_adr].copy()
        arm_qv = self.data.qvel[self.arm_dof_adr].copy()
        grip_q  = np.array([self.data.qpos[self.right_driver_qpos_adr]])
        grip_qv = np.array([self.data.qvel[self.right_driver_dof_adr]])
        tip   = self.data.site_xpos[self.pinch_site_id].copy()
        cube  = self._cube_pos()
        box   = self._box_pos()
        return np.concatenate([
            arm_q, arm_qv, grip_q, grip_qv,
            tip, cube, box, cube - tip, box - cube,
        ]).astype(np.float32)

    def _contact_state(self):
        """Inspect MuJoCo contacts to see what's touching the cube.

        Returns (is_grasping, is_arm_pushing, left_touch, right_touch).
          is_grasping     = at least one LEFT pad and at least one RIGHT pad
                            are in contact with the cube.
          is_arm_pushing  = some non-pad geom is in contact with the cube.
        """
        c_id = self.cube_geom_id
        left_touch = False
        right_touch = False
        non_pad_touch = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == c_id:
                other = c.geom2
            elif c.geom2 == c_id:
                other = c.geom1
            else:
                continue
            if other in self.left_pad_ids:
                left_touch = True
            elif other in self.right_pad_ids:
                right_touch = True
            else:
                # floor / box walls don't count as "arm pushing"
                geom_body = self.model.geom_bodyid[other]
                if geom_body not in (0, self.box_bid):  # 0 == worldbody (floor)
                    non_pad_touch = True
        return (left_touch and right_touch), non_pad_touch, left_touch, right_touch

    def _grasp_state(self):
        is_grasping, arm_pushing, _, _ = self._contact_state()
        return is_grasping, arm_pushing

    def _is_in_box(self, cube_pos: np.ndarray, box_pos: np.ndarray) -> bool:
        dx = abs(cube_pos[0] - box_pos[0])
        dy = abs(cube_pos[1] - box_pos[1])
        z  = cube_pos[2]
        return bool(
            dx < BOX_INNER_HALF
            and dy < BOX_INNER_HALF
            and BOX_FLOOR_TOP_Z - 0.005 < z < BOX_RIM_HEIGHT
        )

    def _sample_workspace_xy(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        if self.curriculum_level >= 2:
            return self.np_random.uniform(low, high)

        center = 0.5 * (low + high)
        span_frac = 0.20 if self.curriculum_level <= 0 else 0.55
        half_span = 0.5 * (high - low) * span_frac
        return self.np_random.uniform(center - half_span, center + half_span)

    def _move_arm_to_target(self, target: np.ndarray):
        for _ in range(90):
            mujoco.mj_forward(self.model, self.data)
            tip = self.data.site_xpos[self.pinch_site_id].copy()
            err = target - tip
            if np.linalg.norm(err) < 0.018:
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.pinch_site_id)
            J = jacp[:, self.arm_dof_adr]
            JJt = J @ J.T + (CART_IK_DAMPING ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, 1.6 * err)
            dq = np.clip(dq, -0.045, 0.045)
            q = self.data.qpos[self.arm_qpos_adr] + dq
            self.data.qpos[self.arm_qpos_adr] = np.clip(
                q, self.ctrl_low[ARM_ACT_SLICE], self.ctrl_high[ARM_ACT_SLICE]
            )

        self.data.ctrl[ARM_ACT_SLICE] = np.clip(
            self.data.qpos[self.arm_qpos_adr],
            self.ctrl_low[ARM_ACT_SLICE],
            self.ctrl_high[ARM_ACT_SLICE],
        )
        self.data.qvel[self.arm_dof_adr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _move_arm_to_hover(self, cube_xy: np.ndarray):
        """Optional reset curriculum: start some episodes near the cube."""
        target = np.array([cube_xy[0], cube_xy[1], CUBE_HALF + HOVER_Z])
        self._move_arm_to_target(target)

    def _move_arm_to_pregrasp(self, cube_pos: np.ndarray):
        """Start close enough that exploration can discover two-pad contact."""
        jitter = self.np_random.uniform(-0.006, 0.006, size=2)
        target = np.array([
            cube_pos[0] + jitter[0],
            cube_pos[1] + jitter[1],
            cube_pos[2] + PRE_GRASP_Z_OFFSET,
        ])
        self._move_arm_to_target(target)

    # =====================================================================
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # --- Arm: a forward / down-pointing ready pose with mild noise -------
        ready = np.array([0.0, -1.6, 1.6, -1.6, -1.57, 0.0])
        ready += self.np_random.uniform(-0.05, 0.05, size=6)
        self.data.qpos[self.arm_qpos_adr] = ready
        self.data.ctrl[ARM_ACT_SLICE] = np.clip(
            ready, self.ctrl_low[ARM_ACT_SLICE], self.ctrl_high[ARM_ACT_SLICE]
        )
        self.data.ctrl[GRIP_ACT_IDX] = 0.0   # gripper open

        # --- Cube: random xy in the left half, resting on the floor ----------
        cube_xy = self._sample_workspace_xy(CUBE_XY_LOW, CUBE_XY_HIGH)
        self.data.qpos[self.cube_qpos_adr + 0] = cube_xy[0]
        self.data.qpos[self.cube_qpos_adr + 1] = cube_xy[1]
        self.data.qpos[self.cube_qpos_adr + 2] = CUBE_HALF + 0.005   # small drop
        # Identity orientation
        self.data.qpos[self.cube_qpos_adr + 3] = 1.0
        self.data.qpos[self.cube_qpos_adr + 4] = 0.0
        self.data.qpos[self.cube_qpos_adr + 5] = 0.0
        self.data.qpos[self.cube_qpos_adr + 6] = 0.0
        # Zero free-joint velocities (6 dofs)
        self.data.qvel[self.cube_dof_adr:self.cube_dof_adr + 6] = 0.0

        # --- Box: random xy in the right half, fixed on the table -----------
        box_xy = self._sample_workspace_xy(BOX_XY_LOW, BOX_XY_HIGH)
        self.data.mocap_pos[self.box_mocap_id] = np.array(
            [box_xy[0], box_xy[1], 0.0]
        )
        self.data.mocap_quat[self.box_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])

        mujoco.mj_forward(self.model, self.data)

        self._reset_start_mode = "home"
        reset_draw = self.np_random.random()
        if reset_draw < self.pregrasp_reset_prob:
            self._move_arm_to_pregrasp(self._cube_pos())
            self._reset_start_mode = "pregrasp"
        elif reset_draw < self.pregrasp_reset_prob + self.near_cube_reset_prob:
            self._move_arm_to_hover(cube_xy)
            self._reset_start_mode = "hover"

        self._step_count = 0
        self._prev_stage = 0
        self._max_stage  = 0
        self._prev_grasp = False
        self._reach_seen = False
        self._left_touch_seen = False
        self._right_touch_seen = False
        self._both_touch_seen = False
        return self._get_obs(), {}

    # =====================================================================
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.action_mode == "joint":
            self._apply_joint_action(action)
        else:
            self._apply_cartesian_action(action)

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, info = self._reward(action)
        self._step_count += 1

        in_box = info["in_box"]
        # Settled-in-box success: cube inside box AND not currently grasped
        # (i.e. it's been released into the tray).
        terminated = bool(in_box and not info["is_grasping"])
        truncated  = self._step_count >= self.max_steps
        info["is_success"] = terminated

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _apply_joint_action(self, action: np.ndarray):
        new_arm = self.data.ctrl[ARM_ACT_SLICE] + action[:6]
        self.data.ctrl[ARM_ACT_SLICE] = np.clip(
            new_arm, self.ctrl_low[ARM_ACT_SLICE], self.ctrl_high[ARM_ACT_SLICE]
        )
        new_grip = float(self.data.ctrl[GRIP_ACT_IDX] + action[6])
        self.data.ctrl[GRIP_ACT_IDX] = float(np.clip(
            new_grip, GRIP_CTRL_MIN, GRIP_CTRL_MAX
        ))

    def _apply_cartesian_action(self, action: np.ndarray):
        delta_xyz = action[:3].astype(np.float64)
        cur_R = self.data.site_xmat[self.pinch_site_id].reshape(3, 3).copy()
        err_r = mat_to_rotvec(self._home_tool_R @ cur_R.T)
        err6 = np.concatenate([
            CART_IK_POS_GAIN * delta_xyz,
            CART_IK_ROT_GAIN * err_r,
        ])

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.pinch_site_id)
        J = np.vstack([jacp, jacr])[:, self.arm_dof_adr]
        JJt = J @ J.T + (CART_IK_DAMPING ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err6)
        dq = np.clip(dq, -ARM_DELTA_LIMIT, ARM_DELTA_LIMIT)

        new_arm = self.data.ctrl[ARM_ACT_SLICE] + dq
        self.data.ctrl[ARM_ACT_SLICE] = np.clip(
            new_arm, self.ctrl_low[ARM_ACT_SLICE], self.ctrl_high[ARM_ACT_SLICE]
        )
        new_grip = float(self.data.ctrl[GRIP_ACT_IDX] + action[3])
        self.data.ctrl[GRIP_ACT_IDX] = float(np.clip(
            new_grip, GRIP_CTRL_MIN, GRIP_CTRL_MAX
        ))

    # =====================================================================
    def _reward(self, action: np.ndarray):
        """Stage-gated absolute-distance reward.

        Stages (only ONE shaping term active per step):
            0  not grasping              → reach: minimize axis-weighted
                                           tip-to-cube distance
            1  grasping, cube not lifted → close the lift gap up to
                                           LIFT_TARGET_Z (saturates above)
            2  grasping, cube lifted     → minimize cube↔box xy distance
            3  cube settled in box       → terminal (no shaping)

        Each stage's shaping is in [-X, 0] — never positive — so the agent
        never gains by loitering. One-shot bonuses fire only the first time
        each stage is reached (latched via `_max_stage`). Each drop outside
        the box pays DROP_PENALTY.
        """
        tip    = self.data.site_xpos[self.pinch_site_id]
        cube   = self._cube_pos()
        box    = self._box_pos()
        cube_z = float(cube[2])

        is_grasping, arm_pushing, left_touch, right_touch = self._contact_state()
        in_box = self._is_in_box(cube, box)
        left_pad = self.data.xpos[self.left_pad_bid]
        right_pad = self.data.xpos[self.right_pad_bid]
        left_pad_dist = float(np.linalg.norm(left_pad - cube))
        right_pad_dist = float(np.linalg.norm(right_pad - cube))
        pad_balance_cost = abs(left_pad_dist - right_pad_dist)

        # ---- Stage selection ----------------------------------------------
        if in_box and not is_grasping:
            stage = 3
        elif is_grasping and cube_z >= LIFT_HEIGHT:
            stage = 2
        elif is_grasping:
            stage = 1
        else:
            stage = 0

        # ---- Per-stage shaping (always ≤ 0) -------------------------------
        if stage == 0:
            xy_tc = float(np.linalg.norm(tip[:2] - cube[:2]))
            z_tc  = float(abs(tip[2] - cube[2]))
            xy_gate = float(np.exp(-((xy_tc / 0.06) ** 2)))
            desired_z = cube[2] + HOVER_Z * (1.0 - xy_gate)
            z_target_err = float(abs(tip[2] - desired_z))

            tool_z = self.data.site_xmat[self.pinch_site_id].reshape(3, 3)[:, 2]
            down_err = float(np.clip(1.0 + tool_z[2], 0.0, 2.0))

            grip_q = float(self.data.qpos[self.right_driver_qpos_adr])
            grip_norm = float(np.clip(grip_q / GRIPPER_Q_MAX, 0.0, 1.0))
            close_window = float(
                np.exp(-((xy_tc / GRASP_WINDOW_XY) ** 2))
                * np.exp(-((z_tc / GRASP_WINDOW_Z) ** 2))
            )
            grip_cost = abs(grip_norm - close_window)

            pad_center = 0.5 * (left_pad + right_pad)
            pad_center_dist = float(np.linalg.norm(pad_center - cube))
            one_sided_contact_cost = 1.0 if (left_touch ^ right_touch) else 0.0

            shape = -(
                W_REACH_XY * xy_tc
                + W_REACH_Z * z_target_err
                + 0.20 * down_err
                + 0.25 * grip_cost
                + 0.35 * close_window * pad_center_dist
                + 0.25 * close_window * pad_balance_cost
                + 0.18 * close_window * right_pad_dist
                + 0.15 * one_sided_contact_cost
            )
        elif stage == 1:
            gap = max(0.0, LIFT_TARGET_Z - cube_z)
            shape = -W_LIFT * gap
        elif stage == 2:
            xy_cb = float(np.linalg.norm(cube[:2] - box[:2]))
            shape = -W_PLACE_XY * xy_cb
        else:
            shape = 0.0

        # ---- One-shot stage advancement bonus (latched) -------------------
        bonus = 0.0
        reach_now = float(np.linalg.norm(tip - cube)) < 0.045
        if stage == 0 and not self._reach_seen and reach_now:
            bonus += REACH_BONUS
            self._reach_seen = True
        if left_touch and not self._left_touch_seen:
            bonus += LEFT_PAD_BONUS
            self._left_touch_seen = True
        if right_touch and not self._right_touch_seen:
            bonus += RIGHT_PAD_BONUS
            self._right_touch_seen = True
        if is_grasping and not self._both_touch_seen:
            bonus += BOTH_PAD_BONUS
            self._both_touch_seen = True
        if stage > self._max_stage:
            if   stage == 1: bonus += GRASP_BONUS
            elif stage == 2: bonus += LIFT_BONUS
            elif stage == 3: bonus += SUCCESS_BONUS
            self._max_stage = stage

        # ---- Drop penalty (each time grasp is lost outside the box) -------
        if self._prev_grasp and not is_grasping and not in_box:
            bonus -= DROP_PENALTY

        # ---- Always-on penalties ------------------------------------------
        push_pen   = PUSH_PENALTY if (arm_pushing and not is_grasping) else 0.0
        norm_action = action / self.action_space.high
        action_pen = 0.01 * float(np.sum(np.square(norm_action)))

        reward = shape + bonus - push_pen - action_pen

        # ---- Update state for next step ----------------------------------
        self._prev_stage = stage
        self._prev_grasp = is_grasping

        info = {
            "stage":       stage,
            "max_stage":   self._max_stage,
            "is_grasping": is_grasping,
            "left_touch":  left_touch,
            "right_touch": right_touch,
            "in_box":      in_box,
            "cube_height": cube_z,
            "place_dist":  float(np.linalg.norm(cube[:2] - box[:2])),
            "reach_dist":  float(np.linalg.norm(tip - cube)),
            "lifted":      bool(self._max_stage >= 2),
            "arm_pushing": arm_pushing,
            "left_pad_dist": left_pad_dist,
            "right_pad_dist": right_pad_dist,
            "pad_balance": pad_balance_cost,
            "curriculum_level": self.curriculum_level,
            "action_mode": self.action_mode,
            "reset_start_mode": self._reset_start_mode,
        }
        return float(reward), info

    # =====================================================================
    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._viewer is None:
                self._viewer = mujoco.Renderer(self.model, height=480, width=640)
            self._viewer.update_scene(self.data)
            return self._viewer.render()
        return None

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except AttributeError:
                pass
            self._viewer = None


if __name__ == "__main__":
    env = PickPlaceEnv()
    obs, _ = env.reset(seed=0)
    print("obs dim:", obs.shape, "action dim:", env.action_space.shape)
    print("cube xyz:", env._cube_pos(), "  box xyz:", env._box_pos())
    print("initial dist (tip -> cube):", np.linalg.norm(
        env.data.site_xpos[env.pinch_site_id] - env._cube_pos()
    ))
    total = 0.0
    info = {}
    for _ in range(50):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    print(f"random rollout reward={total:.2f}, "
          f"place_dist={info.get('place_dist', 0):.3f}, "
          f"in_box={info.get('in_box', False)}")
    env.close()
