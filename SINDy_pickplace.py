import os
import time

import numpy as np
import mujoco
import pandas as pd
# Use the non-interactive Agg backend: we only write PNGs (never plt.show()),
# and on macOS the default GUI backend conflicts with mjpython's main-thread
# ownership when we also open a live MuJoCo viewer.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pickplace_env import (
    PickPlaceEnv,
    ARM_ACT_SLICE, GRIP_ACT_IDX,
    GRIP_CTRL_MIN, GRIP_CTRL_MAX,
)


# ============================================================================
# CONFIG
# ============================================================================
RENDER_MODE = "viewer"      # "viewer" | "video" | "none"
EPISODES    = 5
SEED_OFFSET = 0
MAX_STEPS   = 600
RESULTS_DIR = "./results/sindy_gd_pickplace" # CHANGED FROM "./results/pid_pickplace" TO REFLECT NEW CONTROLLER 

# --- SINDy CONFIG ----------------------------------------------------
# SINDy controller parameters
SINDY_PARAMS = dict(
    coeffs=None,
    max_dq=0.07,
    damping=0.05,
    reset_on_phase=True,
    lr=1e-4,
    train=True,
)



# --- Waypoint heights -------------------------------------------------------
APPROACH_Z   = 0.20
PRE_GRASP_Z  = 0.05
GRASP_Z      = 0.015
LIFT_Z       = 0.25
RELEASE_Z    = 0.12

# --- Phase advancement ------------------------------------------------------
POS_TOL              = 0.012
DESCEND_Z_TOL        = 0.005
GRIPPER_CLOSED_Q     = 0.40
GRIPPER_OPEN_Q       = 0.05
GRASP_HOLD_STEPS     = 20
RELEASE_HOLD_STEPS   = 30
PHASE_TIMEOUT        = 200

VIDEO_DIR = "./videos_sindy" # CHANGED FROM "./videos_pid" TO REFLECT NEW CONTROLLER
VIDEO_FPS  = 50
STEP_DELAY = 0.02
# ============================================================================


PHASES = ["APPROACH", "PRE_GRASP", "DESCEND", "GRASP",
          "LIFT", "TRANSPORT", "RELEASE", "DONE"]


# ----------------------------------------------------------------------------
# Math helper
# ----------------------------------------------------------------------------
# helps convert orientation error to rotation vector
def mat_to_rotvec(R: np.ndarray) -> np.ndarray:
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

# -----------------Class for SINDy controller replacing PID-------------------------------------
class SINDyController:
    def __init__(self, coeffs=None, max_dq=0.07, damping=0.05,
                 reset_on_phase=True, lr=1e-4, train=True):
        self.max_dq = float(max_dq)
        self.damping = float(damping)
        self.reset_on_phase = bool(reset_on_phase)
        self.lr = float(lr)
        self.train = bool(train)

        self.n_features = 6*3  # error, sign(error), error^2 for each of the 6 error dimensions

        if coeffs is None:
            self.coeffs = np.random.uniform(-0.01, 0.01, size=(self.n_features, 6))
        else:
            self.coeffs = np.array(coeffs, dtype=np.float64)
        self.reset()

    def reset(self):
        self._last_theta = None
        self._last_u = None
        self._last_e = None

    def on_phase_change(self):
        if self.reset_on_phase:
            self.reset()

    def build_library(self, e):
        theta = np.concatenate([
            e,              # error
            np.sign(e),     # direction of error
            e ** 2          # squared error
        ])

        return theta
        

    def compute(self, model, data, site_id, arm_dof_adr, tgt_pos, tgt_R):
        cur_pos = data.site_xpos[site_id].copy()
        cur_R = data.site_xmat[site_id].reshape(3, 3).copy()

        e_pos = tgt_pos - cur_pos
        e_rot = mat_to_rotvec(tgt_R @ cur_R.T)
        e = np.concatenate([e_pos, e_rot])

        theta = self.build_library(e)

        u = theta @ self.coeffs

        self._last_theta = theta.copy()
        self._last_u = u.copy()
        self._last_e = e.copy()

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        J = np.vstack([jacp, jacr])[:, arm_dof_adr]
        JJt = J @ J.T + (self.damping ** 2) * np.eye(6)

        dq = J.T @ np.linalg.solve(JJt, u)

        return np.clip(dq, -self.max_dq, self.max_dq), e

    def update_from_system_output(self, new_error):
        if not self.train:
            return

        if self._last_theta is None or self._last_u is None or self._last_e is None:
            return

        desired_error_reduction = self._last_e - new_error

        target_u = desired_error_reduction

        pred_u = self._last_u

        error = pred_u - target_u

        grad = np.outer(self._last_theta, error)

        self.coeffs -= self.lr * grad

# ----------------------------------------------------------------------------
# Waypoint plan
# ----------------------------------------------------------------------------
def waypoint(phase, cube_xy, box_xy):
    cx, cy = cube_xy
    bx, by = box_xy
    if phase == "APPROACH":  return np.array([cx, cy, APPROACH_Z]),  GRIP_CTRL_MIN
    if phase == "PRE_GRASP": return np.array([cx, cy, PRE_GRASP_Z]), GRIP_CTRL_MIN
    if phase == "DESCEND":   return np.array([cx, cy, GRASP_Z]),     GRIP_CTRL_MIN
    if phase == "GRASP":     return np.array([cx, cy, GRASP_Z]),     GRIP_CTRL_MAX
    if phase == "LIFT":      return np.array([cx, cy, LIFT_Z]),      GRIP_CTRL_MAX
    if phase == "TRANSPORT": return np.array([bx, by, LIFT_Z]),      GRIP_CTRL_MAX
    if phase == "RELEASE":   return np.array([bx, by, RELEASE_Z]),   GRIP_CTRL_MIN
    return np.array([bx, by, RELEASE_Z]), GRIP_CTRL_MIN


def compute_home_R(env):
    env.data.qpos[env.arm_qpos_adr] = np.array(
        [0.0, -1.6, 1.6, -1.6, -1.57, 0.0]
    )
    mujoco.mj_forward(env.model, env.data)
    return env.data.site_xmat[env.pinch_site_id].reshape(3, 3).copy()


# ----------------------------------------------------------------------------
# Episode runner with per-step logging
# ----------------------------------------------------------------------------
def run_episode(env, controller, seed, target_R, save_frames=False):
    obs, _ = env.reset(seed=seed)
    controller.reset()

    model, data = env.model, env.data
    site_id  = env.pinch_site_id
    arm_qadr = env.arm_qpos_adr
    arm_dadr = env.arm_dof_adr

    cube_xy = env._cube_pos()[:2].copy()
    box_xy  = env._box_pos()[:2].copy()

    phase_idx = 0
    phase_steps = 0
    frames = []
    last_info = {}
    total_reward = 0.0
    arm_act_high  = float(env.action_space.high[0])
    grip_act_high = float(env.action_space.high[6])

    log_rows = []

    for t in range(env.max_steps):
        phase = PHASES[phase_idx]
        target_pos, grip_ctrl = waypoint(phase, cube_xy, box_xy)
        dq, err6 = controller.compute(model, data, site_id, arm_dadr,
                                      target_pos, target_R)

        new_ctrl_arm = data.qpos[arm_qadr] + dq
        delta_arm    = new_ctrl_arm - data.ctrl[ARM_ACT_SLICE]
        delta_grip   = grip_ctrl - data.ctrl[GRIP_ACT_IDX]
        action = np.zeros(7, dtype=np.float32)
        action[:6] = np.clip(delta_arm, -arm_act_high, arm_act_high)
        action[6]  = float(np.clip(delta_grip, -grip_act_high, grip_act_high))

        obs, reward, term, trunc, info = env.step(action)

        # -----------GD UPDATE FOR SINDy CONTROLLER BASED ON SYSTEM OUTPUT ERROR--------------------
        new_tip = data.site_xpos[site_id].copy()
        new_R = data.site_xmat[site_id].reshape(3, 3).copy()

        new_e_pos = target_pos - new_tip
        new_e_rot = mat_to_rotvec(target_R @ new_R.T)
        new_err6 = np.concatenate([new_e_pos, new_e_rot])

        if hasattr(controller, "update_from_system_output"):
            controller.update_from_system_output(new_err6)
        # ------------------------------------------------------------------------------------------

        total_reward += reward
        last_info = info
        phase_steps += 1

        if save_frames:
            frames.append(env.render())
        if RENDER_MODE == "viewer":
            time.sleep(STEP_DELAY)

        # --- log ---
        cube_now = env._cube_pos()
        tip = data.site_xpos[site_id].copy()
        log_rows.append({
            "step": t, "phase": phase,
            "tgt_x": target_pos[0], "tgt_y": target_pos[1], "tgt_z": target_pos[2],
            "tip_x": tip[0], "tip_y": tip[1], "tip_z": tip[2],
            "cube_x": cube_now[0], "cube_y": cube_now[1], "cube_z": cube_now[2],
            "pos_err": float(np.linalg.norm(err6[:3])),
            "rot_err": float(np.linalg.norm(err6[3:])),
            "grip_q": float(data.qpos[env.right_driver_qpos_adr]),
            "reward": float(reward),
            **{f"q{i}": float(data.qpos[arm_qadr[i]]) for i in range(6)},
        })

        # --- phase advancement ---
        xy_err = float(np.linalg.norm(tip[:2] - target_pos[:2]))
        z_err  = abs(tip[2] - target_pos[2])
        pos_err = float(np.linalg.norm(tip - target_pos))
        grip_q = float(data.qpos[env.right_driver_qpos_adr])

        advance = False
        if   phase == "APPROACH":  advance = pos_err < POS_TOL
        elif phase == "PRE_GRASP": advance = xy_err < 0.008 and z_err < 0.005
        elif phase == "DESCEND":   advance = z_err < DESCEND_Z_TOL and xy_err < 0.012
        elif phase == "GRASP":     advance = grip_q > GRIPPER_CLOSED_Q and phase_steps > GRASP_HOLD_STEPS
        elif phase == "LIFT":      advance = z_err < 0.02
        elif phase == "TRANSPORT": advance = xy_err < POS_TOL
        elif phase == "RELEASE":   advance = grip_q < GRIPPER_OPEN_Q and phase_steps > RELEASE_HOLD_STEPS

        if advance or phase_steps > PHASE_TIMEOUT:
            phase_idx = min(phase_idx + 1, len(PHASES) - 1)
            phase_steps = 0
            controller.on_phase_change()

        if term or trunc:
            break

    return total_reward, last_info, frames, pd.DataFrame(log_rows)


# ----------------------------------------------------------------------------
# Aggregations
# ----------------------------------------------------------------------------
def summarize(df, success, total_reward, last_info):
    per_phase = df.groupby("phase", sort=False).agg(
        steps=("step", "count"),
        mean_pos_err=("pos_err", "mean"),
        max_pos_err=("pos_err", "max"),
        mean_rot_err=("rot_err", "mean"),
    ).reset_index()
    summary = {
        "success":      int(success),
        "total_steps":  len(df),
        "total_reward": float(total_reward),
        "place_dist":   float(last_info.get("place_dist", np.nan)),
        "max_pos_err":  float(df["pos_err"].max()),
        "mean_pos_err": float(df["pos_err"].mean()),
        "rms_pos_err":  float(np.sqrt((df["pos_err"] ** 2).mean())),
    }
    return summary, per_phase


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
PHASE_COLORS = {
    "APPROACH":  "#cce5ff",
    "PRE_GRASP": "#a3d8ff",
    "DESCEND":   "#80c0ff",
    "GRASP":     "#ffd9b3",
    "LIFT":      "#c2f0c2",
    "TRANSPORT": "#a3e5a3",
    "RELEASE":   "#ffb3b3",
    "DONE":      "#dddddd",
}


def shade_phases(ax, df):
    if df.empty:
        return
    cur = df["phase"].iloc[0]
    start = df["step"].iloc[0]
    for i in range(1, len(df)):
        if df["phase"].iloc[i] != cur:
            ax.axvspan(start, df["step"].iloc[i],
                       color=PHASE_COLORS.get(cur, "#eee"), alpha=0.4, lw=0)
            cur = df["phase"].iloc[i]
            start = df["step"].iloc[i]
    ax.axvspan(start, df["step"].iloc[-1],
               color=PHASE_COLORS.get(cur, "#eee"), alpha=0.4, lw=0)


def plot_episode(df, ep, plots_dir):
    fig = plt.figure(figsize=(14, 10))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(df["tip_x"], df["tip_y"], df["tip_z"], lw=1.2, color="C0", label="TCP")
    ax3d.plot(df["tgt_x"], df["tgt_y"], df["tgt_z"], lw=1.0, color="C3",
              ls="--", alpha=0.7, label="target")
    ax3d.scatter([df["cube_x"].iloc[0]], [df["cube_y"].iloc[0]], [df["cube_z"].iloc[0]],
                 color="black", s=40, label="cube start")
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title(f"Episode {ep} — TCP trajectory")
    ax3d.legend(loc="best", fontsize=8)

    ax1 = fig.add_subplot(2, 2, 2)
    shade_phases(ax1, df)
    ax1.plot(df["step"], df["pos_err"], color="C0", lw=1.2, label="|pos err| (m)")
    ax1.plot(df["step"], df["rot_err"], color="C3", lw=1.0, label="|rot err| (rad)")
    ax1.set_xlabel("step"); ax1.set_ylabel("error")
    ax1.set_title("Tracking error (background = phase)")
    ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 3)
    shade_phases(ax2, df)
    ax2.plot(df["step"], df["cube_z"], color="C2", lw=1.5, label="cube z (m)")
    ax2.plot(df["step"], df["tip_z"], color="C0", lw=0.8, ls=":", label="tip z (m)")
    ax2t = ax2.twinx()
    ax2t.plot(df["step"], df["grip_q"], color="C1", lw=1.0, label="grip q (rad)")
    ax2t.set_ylabel("grip q (rad)", color="C1")
    ax2.set_xlabel("step"); ax2.set_ylabel("height (m)")
    ax2.set_title("Cube height & gripper state")
    ax2.legend(loc="upper left", fontsize=8); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 4)
    shade_phases(ax3, df)
    for i in range(6):
        ax3.plot(df["step"], df[f"q{i}"], lw=1.0, label=f"q{i}")
    ax3.set_xlabel("step"); ax3.set_ylabel("joint angle (rad)")
    ax3.set_title("Arm joint trajectories")
    ax3.legend(loc="best", fontsize=7, ncol=3); ax3.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(plots_dir, f"episode_{ep}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def plot_aggregate(summary_df, phase_df, plots_dir):
    # Place-distance histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(summary_df["place_dist"].dropna(), bins=10,
            color="C0", edgecolor="black")
    ax.axvline(0.06, color="C3", ls="--",
               label="box inner half-extent (0.06 m)")
    ax.set_xlabel("place distance (cube center -> box center, m)")
    ax.set_ylabel("episodes")
    ax.set_title(f"Final placement accuracy ({len(summary_df)} episodes)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(plots_dir, "place_dist_hist.png")
    fig.savefig(p1, dpi=130); plt.close(fig)

    # Per-phase aggregate
    agg = phase_df.groupby("phase", sort=False).agg(
        mean_steps=("steps", "mean"),
        std_steps=("steps", "std"),
        mean_pos_err=("mean_pos_err", "mean"),
        mean_max_pos_err=("max_pos_err", "mean"),
    ).reindex([p for p in PHASES if p != "DONE"]).dropna(how="all")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ax = axes[0]
    ax.bar(agg.index, agg["mean_steps"],
           yerr=agg["std_steps"].fillna(0),
           color=[PHASE_COLORS.get(p, "#888") for p in agg.index],
           edgecolor="black")
    ax.set_ylabel("steps"); ax.set_title("Mean phase duration ± std")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    ax = axes[1]
    width = 0.4
    x = np.arange(len(agg.index))
    ax.bar(x - width/2, agg["mean_pos_err"], width=width,
           color="C0", label="mean pos err", edgecolor="black")
    ax.bar(x + width/2, agg["mean_max_pos_err"], width=width,
           color="C3", label="max pos err", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=20, ha="right")
    ax.set_ylabel("error (m)"); ax.set_title("Per-phase tracking error")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    p2 = os.path.join(plots_dir, "phase_summary.png")
    fig.savefig(p2, dpi=130); plt.close(fig)
    return [p1, p2]


# ----------------------------------------------------------------------------
# === SWAP THIS LINE TO REPLACE THE CONTROLLER =========================
# Any object implementing .reset() / .on_phase_change() /
# .compute(model, data, site_id, arm_dof_adr, tgt_pos, tgt_R) -> (dq, err6)
# will plug in.
# ============================================================================
CONTROLLER = SINDyController(**SINDY_PARAMS)

def main():
    if RENDER_MODE == "viewer":
        env = PickPlaceEnv(render_mode="human", max_steps=MAX_STEPS)
    elif RENDER_MODE == "video":
        env = PickPlaceEnv(render_mode="rgb_array", max_steps=MAX_STEPS)
        os.makedirs(VIDEO_DIR, exist_ok=True)
    else:
        env = PickPlaceEnv(render_mode=None, max_steps=MAX_STEPS)

    env.reset(seed=0)
    target_R = compute_home_R(env)

    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    summary_rows = []
    phase_rows   = []
    start_train_time = time.perf_counter()

    for ep in range(EPISODES):
        print(f"\n=== episode {ep} ===")
        ret, info, frames, df = run_episode(
            env, CONTROLLER, SEED_OFFSET + ep, target_R,
            save_frames=(RENDER_MODE == "video"),
        )
        success = bool(info.get("is_success", False))
        summary, per_phase = summarize(df, success, ret, info)
        summary["episode"] = ep
        summary_rows.append(summary)
        per_phase["episode"] = ep
        phase_rows.append(per_phase)

        # Save SINDy trajectory for evaluation
        os.makedirs("results/trajectories", exist_ok=True)

        trajectory_df = df[["step", "tip_x", "tip_y", "tip_z", "reward"]].copy()
        trajectory_df.insert(0, "episode", ep + 1)

        trajectory_df = trajectory_df.rename(columns={
            "tip_x": "actual_x",
            "tip_y": "actual_y",
            "tip_z": "actual_z"
        })

        trajectory_path = "results/trajectories/SINDy_trajectory.csv"

        if ep == 0:
            trajectory_df.to_csv(trajectory_path, index=False)
        else:
            trajectory_df.to_csv(trajectory_path, mode="a", header=False, index=False)

        log_path = trajectory_path
        #-------------------------------------------------------------------------------------

        plot_path = plot_episode(df, ep, plots_dir)

        print(f"  return={ret:7.2f}  success={success}  "
              f"place_dist={summary['place_dist']:.3f}  "
              f"max_pos_err={summary['max_pos_err']:.3f}")
        print(f"  log={log_path}  plot={plot_path}")

        if RENDER_MODE == "video" and frames:
            try:
                import imageio.v2 as imageio
                vp = os.path.join(VIDEO_DIR, f"sindy_ep{ep}.mp4")
                imageio.mimsave(vp, frames, fps=VIDEO_FPS, macro_block_size=1)
                print(f"  saved {vp}")
            except Exception as e:
                print(f"  [video] failed: {e}")

    env.close()

    summary_df = pd.DataFrame(summary_rows).set_index("episode")
    phase_df   = pd.concat(phase_rows, ignore_index=True)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "episode_summary.csv"))
    phase_df.to_csv(os.path.join(RESULTS_DIR, "phase_summary.csv"), index=False)

    plot_aggregate(summary_df, phase_df, plots_dir)

    print("\n" + "=" * 78)
    print("EPISODE SUMMARY")
    print("=" * 78)
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))
    n_succ = int(summary_df["success"].sum())
    n      = len(summary_df)
    print(f"\nsuccess: {n_succ}/{n}    "
          f"mean return: {summary_df['total_reward'].mean():.2f}    "
          f"mean place_dist: {summary_df['place_dist'].mean():.4f} m    "
          f"mean RMS pos err: {summary_df['rms_pos_err'].mean():.4f} m")

    # Aggregate per-phase across episodes (printed table)
    print("\n" + "=" * 78)
    print("PER-PHASE AGGREGATE (mean across episodes)")
    print("=" * 78)
    agg = phase_df.groupby("phase", sort=False).agg(
        mean_steps=("steps", "mean"),
        std_steps=("steps", "std"),
        mean_pos_err=("mean_pos_err", "mean"),
        mean_max_pos_err=("max_pos_err", "mean"),
        mean_rot_err=("mean_rot_err", "mean"),
    ).reindex([p for p in PHASES if p != "DONE"]).dropna(how="all")
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))

    end_train_time = time.perf_counter()
    training_time = end_train_time - start_train_time

    with open(os.path.join(RESULTS_DIR, "training_time.txt"), "w") as f:
        f.write(str(training_time))

    print(f"SINDy training/optimization time: {training_time:.2f} seconds")
    print(f"\nresults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

    if isinstance(CONTROLLER, SINDyController):
        np.save("sindy_gd_coeffs.npy", CONTROLLER.coeffs)
        print("Saved GD-updated SINDy coefficients to sindy_gd_coeffs.npy")