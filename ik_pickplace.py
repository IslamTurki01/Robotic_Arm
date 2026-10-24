"""Scripted IK + waypoint controller for the pick-and-place task.

A non-learned baseline. Uses Jacobian-based differential inverse kinematics
(damped least squares) to drive the pinch site through a sequence of
Cartesian waypoints, opening/closing the Robotiq fingers between phases.
The arm actuators in robot_pickplace.xml are already position-PD, so this
script provides only the *outer* Cartesian P-controller — no extra PID layer
on top.

Run with:
    mjpython ik_pickplace.py        # live viewer (macOS)
    python3 ik_pickplace.py         # headless / video — set RENDER_MODE below

Phases:
    APPROACH   pinch hovers above the cube, gripper open
    DESCEND    pinch drops to cube-center height, gripper open
    GRASP      hold position, close gripper
    LIFT       raise the cube clear of the table
    TRANSPORT  move horizontally over the box
    RELEASE    descend slightly, open gripper to drop the cube
"""

import os
import time

import numpy as np
import pandas as pd
import mujoco

from pickplace_env import (
    PickPlaceEnv,
    ARM_ACT_SLICE, GRIP_ACT_IDX,
    GRIP_CTRL_MIN, GRIP_CTRL_MAX,
)

# ============================================================
# CONFIG
# ============================================================
RENDER_MODE = "viewer"      # "viewer" | "video" | "none"
EPISODES    = 3
SEED_OFFSET = 0
STEP_DELAY  = 0.02          # wall-clock pacing for the live viewer

MAX_STEPS = 600             # episode length cap for the IK controller

# IK / Cartesian P-controller gains
KP_POS    = 2.5             # P-gain on TCP position error
KP_ROT    = 1.0             # P-gain on TCP orientation error
MAX_DQ    = 0.07            # max per-step joint-target change (rad)
DAMPING   = 0.05            # damped-least-squares regularizer (rad·s/m)

# Waypoint heights (world-frame z of the pinch site)
APPROACH_Z  = 0.20
PRE_GRASP_Z = 0.05          # just above cube top (3 cm cube → top at 0.03)
GRASP_Z     = 0.015         # pinch at cube-center height
LIFT_Z      = 0.25
RELEASE_Z   = 0.12

# Phase advancement
POS_TOL              = 0.012
DESCEND_Z_TOL        = 0.005
GRIPPER_CLOSED_Q     = 0.40
GRIPPER_OPEN_Q       = 0.05
GRASP_HOLD_STEPS     = 20
RELEASE_HOLD_STEPS   = 30
PHASE_TIMEOUT        = 200

VIDEO_DIR = "./videos_ik"
VIDEO_FPS = 50
# ============================================================


PHASES = ["APPROACH", "PRE_GRASP", "DESCEND", "GRASP",
          "LIFT", "TRANSPORT", "RELEASE", "DONE"]


def mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector (Rodrigues, world frame)."""
    cos_t = np.clip((R[0, 0] + R[1, 1] + R[2, 2] - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_t))
    if theta < 1e-6:
        return np.zeros(3)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))
    return axis * theta


def ik_delta(model, data, site_id, arm_dof_adr,
             target_pos, target_R) -> np.ndarray:
    """Damped least-squares 6-DOF IK step for the arm.

    Returns a length-6 Δq (rad) such that one kinematic step toward
    (cur + Δq) reduces the (pos, rot) error of the pinch site.
    """
    cur_pos = data.site_xpos[site_id].copy()
    cur_R   = data.site_xmat[site_id].reshape(3, 3).copy()

    err_p = KP_POS * (target_pos - cur_pos)
    err_r = KP_ROT * mat_to_rotvec(target_R @ cur_R.T)
    err6  = np.concatenate([err_p, err_r])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp, jacr])              # (6, nv)
    J_arm = J[:, arm_dof_adr]                # (6, 6)

    JJt = J_arm @ J_arm.T + (DAMPING ** 2) * np.eye(6)
    dq  = J_arm.T @ np.linalg.solve(JJt, err6)
    return np.clip(dq, -MAX_DQ, MAX_DQ)


def waypoint(phase: str, cube_xy: np.ndarray, box_xy: np.ndarray):
    """(target_xyz, gripper_ctrl) for the given phase."""
    cx, cy = cube_xy
    bx, by = box_xy
    if phase == "APPROACH":
        return np.array([cx, cy, APPROACH_Z]),  GRIP_CTRL_MIN
    if phase == "PRE_GRASP":
        return np.array([cx, cy, PRE_GRASP_Z]), GRIP_CTRL_MIN
    if phase == "DESCEND":
        return np.array([cx, cy, GRASP_Z]),     GRIP_CTRL_MIN
    if phase == "GRASP":
        return np.array([cx, cy, GRASP_Z]),    GRIP_CTRL_MAX
    if phase == "LIFT":
        return np.array([cx, cy, LIFT_Z]),     GRIP_CTRL_MAX
    if phase == "TRANSPORT":
        return np.array([bx, by, LIFT_Z]),     GRIP_CTRL_MAX
    if phase == "RELEASE":
        return np.array([bx, by, RELEASE_Z]),  GRIP_CTRL_MIN
    return np.array([bx, by, RELEASE_Z]),      GRIP_CTRL_MIN


def compute_home_R(env: PickPlaceEnv) -> np.ndarray:
    """Snapshot pinch orientation at the un-noisy home pose.

    We then keep this orientation as the IK target throughout the trajectory
    so the gripper stays pointed straight down for top-down grasping.
    """
    env.data.qpos[env.arm_qpos_adr] = np.array(
        [0.0, -1.6, 1.6, -1.6, -1.57, 0.0]
    )
    mujoco.mj_forward(env.model, env.data)
    return env.data.site_xmat[env.pinch_site_id].reshape(3, 3).copy()


def run_episode(env: PickPlaceEnv, seed: int, target_R: np.ndarray,
                save_frames: bool):
    obs, _ = env.reset(seed=seed)
    model, data = env.model, env.data
    site_id  = env.pinch_site_id
    arm_qadr = env.arm_qpos_adr
    arm_dadr = env.arm_dof_adr

    # Snapshot cube/box xy once at episode start. The cube doesn't move on
    # its own, and chasing a live cube position causes the gripper to keep
    # adjusting target xy while it's bumping the cube during descent —
    # which is exactly how grasps fail.
    cube_xy = env._cube_pos()[:2].copy()
    box_xy  = env._box_pos()[:2].copy()
    print(f"  cube xy={cube_xy}  box xy={box_xy}")

    phase_idx     = 0
    phase_steps   = 0
    frames        = []
    last_info     = {}
    total_reward  = 0.0

    arm_act_high  = float(env.action_space.high[0])
    grip_act_high = float(env.action_space.high[6])

    trajectory_data = []
    for step in range(env.max_steps):

        phase = PHASES[phase_idx]
        target_pos, grip_ctrl = waypoint(phase, cube_xy, box_xy)

        dq = ik_delta(model, data, site_id, arm_dadr, target_pos, target_R)

        # Convert "absolute desired ctrl" into the env's delta action
        new_ctrl_arm = data.qpos[arm_qadr] + dq
        delta_arm = new_ctrl_arm - data.ctrl[ARM_ACT_SLICE]
        delta_grip = grip_ctrl - data.ctrl[GRIP_ACT_IDX]

        action = np.zeros(7, dtype=np.float32)
        action[:6] = np.clip(delta_arm, -arm_act_high, arm_act_high)
        action[6]  = float(np.clip(delta_grip, -grip_act_high, grip_act_high))

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        last_info = info
        phase_steps += 1
        if save_frames:
            frames.append(env.render())
        if RENDER_MODE == "viewer":
            time.sleep(STEP_DELAY)

        # Phase-advancement logic
        tip     = data.site_xpos[site_id].copy()
        trajectory_data.append({
            "episode": seed - SEED_OFFSET + 1,
            "step": step,
            "actual_x": tip[0],
            "actual_y": tip[1],
            "actual_z": tip[2],
            "reward": reward,
            "is_success": int(bool(info.get("is_success", False))),
            "place_dist": float(info.get("place_dist", np.nan))
        })
        pos_err = float(np.linalg.norm(tip - target_pos))
        grip_q  = float(data.qpos[env.right_driver_qpos_adr])

        xy_err = float(np.linalg.norm(tip[:2] - target_pos[:2]))
        z_err  = abs(tip[2] - target_pos[2])

        advance = False
        if phase == "APPROACH":
            advance = pos_err < POS_TOL
        elif phase == "PRE_GRASP":
            # Insist on tight xy AND z before committing to the final descent.
            advance = xy_err < 0.008 and z_err < 0.005
        elif phase == "DESCEND":
            advance = z_err < DESCEND_Z_TOL and xy_err < 0.012
        elif phase == "GRASP":
            advance = (grip_q > GRIPPER_CLOSED_Q
                       and phase_steps > GRASP_HOLD_STEPS)
        elif phase == "LIFT":
            advance = abs(tip[2] - target_pos[2]) < 0.02
        elif phase == "TRANSPORT":
            xy_err = float(np.linalg.norm(tip[:2] - target_pos[:2]))
            advance = xy_err < POS_TOL
        elif phase == "RELEASE":
            advance = (grip_q < GRIPPER_OPEN_Q
                       and phase_steps > RELEASE_HOLD_STEPS)

        if advance or phase_steps > PHASE_TIMEOUT:
            print(f"  [phase] {phase} -> {PHASES[min(phase_idx+1, len(PHASES)-1)]}"
                  f"  (steps={phase_steps}, pos_err={pos_err:.3f})")
            phase_idx = min(phase_idx + 1, len(PHASES) - 1)
            phase_steps = 0

        if term or trunc:
            break

    return total_reward, last_info, frames, trajectory_data


def main():
    if RENDER_MODE == "viewer":
        env = PickPlaceEnv(render_mode="human", max_steps=MAX_STEPS)
    elif RENDER_MODE == "video":
        env = PickPlaceEnv(render_mode="rgb_array", max_steps=MAX_STEPS)
        os.makedirs(VIDEO_DIR, exist_ok=True)
    else:
        env = PickPlaceEnv(render_mode=None, max_steps=MAX_STEPS)

    # One reset to construct the data, then snapshot the home pose orientation.
    env.reset(seed=0)
    target_R = compute_home_R(env)

    n_succ = 0
    returns = []
    all_trajectory_data = []
    for ep in range(EPISODES):
        print(f"\n=== episode {ep} ===")
        ret, info, frames, trajectory_data = run_episode(
            env, SEED_OFFSET + ep, target_R,
            save_frames=(RENDER_MODE == "video"),
        )
        returns.append(ret)
        all_trajectory_data.extend(trajectory_data)
        ok = bool(info.get("is_success", False))
        n_succ += int(ok)
        print(f"  return={ret:7.2f}  in_box={info.get('in_box', False)}  "
              f"success={ok}  place_dist={info.get('place_dist', 0):.3f}")

        if RENDER_MODE == "video" and frames:
            try:
                import imageio.v2 as imageio
                path = os.path.join(VIDEO_DIR, f"ik_ep{ep}.mp4")
                imageio.mimsave(path, frames, fps=VIDEO_FPS, macro_block_size=1)
                print(f"  saved {path}")
            except Exception as e:
                print(f"  [video] failed: {e}")
    os.makedirs("results/trajectories", exist_ok=True)
    pd.DataFrame(all_trajectory_data).to_csv(
        "results/trajectories/IK_trajectory.csv",
        index=False
    )
    print("Saved: results/trajectories/IK_trajectory.csv")
    env.close()
    print(f"\nIK controller: success {n_succ}/{EPISODES}, "
          f"mean return {np.mean(returns):.2f}")


if __name__ == "__main__":
    main()
