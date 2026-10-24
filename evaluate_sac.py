import glob
import os
import time
import pandas as pd

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pickplace_env import PickPlaceEnv

MODEL_PATHS = [
    "./runs/sac_pickplace_fast/final_model.zip",
]

RENDER_MODE = "viewer"

EPISODES       = 5
STEP_DELAY     = 0.02
DETERMINISTIC  = True
SEED_OFFSET    = 0
MAX_STEPS      = 200
VIDEO_DIR      = "./videos_pickplace"
VIDEO_FPS      = 50


def _save_video(frames, path, fps):
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        return path
    except Exception as e:
        print(f"[video] imageio failed ({e}); falling back to GIF")
        from PIL import Image
        gif_path = path.rsplit(".", 1)[0] + ".gif"
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
        return gif_path


def expand_paths(patterns):
    seen, out = set(), []
    for p in patterns:
        matches = sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
        for m in matches:
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def infer_action_mode(model):
    action_dim = int(np.prod(model.action_space.shape))
    if action_dim == 4:
        return "cartesian"
    if action_dim == 7:
        return "joint"
    raise ValueError(f"Unsupported pick-place action dim: {action_dim}")


def uses_normalized_actions(model):
    low = np.asarray(model.action_space.low)
    high = np.asarray(model.action_space.high)
    return np.allclose(low, -1.0) and np.allclose(high, 1.0)


def find_vecnormalize_path(model_path):
    model_dir = os.path.dirname(model_path)
    candidates = [
        os.path.join(model_dir, "vecnormalize.pkl"),
        os.path.join(os.path.dirname(model_dir), "vecnormalize.pkl"),
        os.path.join(os.path.dirname(os.path.dirname(model_dir)), "vecnormalize.pkl"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def make_raw_env(action_mode):
    if RENDER_MODE == "viewer":
        return PickPlaceEnv(render_mode="human", max_steps=MAX_STEPS, action_mode=action_mode)
    if RENDER_MODE == "video":
        return PickPlaceEnv(render_mode="rgb_array", max_steps=MAX_STEPS, action_mode=action_mode)
    return PickPlaceEnv(render_mode=None, max_steps=MAX_STEPS, action_mode=action_mode)


def rescale_to_unit_action(env):
    low = np.full(env.action_space.shape, -1.0, dtype=np.float32)
    high = np.full(env.action_space.shape, 1.0, dtype=np.float32)
    return gym.wrappers.RescaleAction(env, low, high)


def make_eval_env(model_path, model):
    action_mode = infer_action_mode(model)
    raw_env = make_raw_env(action_mode)
    env = raw_env

    normalized_actions = uses_normalized_actions(model)
    if normalized_actions:
        env = rescale_to_unit_action(raw_env)

    norm_path = find_vecnormalize_path(model_path)
    if not norm_path:
        return env, raw_env, False, action_mode, normalized_actions, None

    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(norm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    return vec_env, raw_env, True, action_mode, normalized_actions, norm_path


def reset_eval_env(env, is_vec, seed):
    if is_vec:
        env.seed(seed)
        return env.reset()
    obs, _ = env.reset(seed=seed)
    return obs


def step_eval_env(env, is_vec, action):
    if is_vec:
        obs, reward, done, infos = env.step(action)
        return obs, float(reward[0]), bool(done[0]), infos[0]
    obs, reward, terminated, truncated, info = env.step(action)
    return obs, float(reward), bool(terminated or truncated), info


def evaluate_one(env, raw_env, is_vec, model, label, save_video):
    successes = 0
    returns = []
    trajectory_data = []
    metrics_data = []

    for ep in range(EPISODES):
        obs = reset_eval_env(env, is_vec, SEED_OFFSET + ep)
        done = False
        ep_ret = 0.0
        frames = []
        info = {}
        step = 0

        while not done:
            start_time = time.perf_counter()
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)
            inference_time = time.perf_counter() - start_time

            obs, reward, done, info = step_eval_env(env, is_vec, action)
            ep_ret += reward

            actual_pos = raw_env.data.site_xpos[raw_env.pinch_site_id].copy()

            trajectory_data.append({
                "episode": ep + 1,
                "step": step,
                "actual_x": actual_pos[0],
                "actual_y": actual_pos[1],
                "actual_z": actual_pos[2],
                "reward": reward,
                "is_success": int(bool(info.get("is_success", False))),
                "place_dist": float(info.get("place_dist", np.nan)),
                "inference_time": inference_time
            })

            if RENDER_MODE == "viewer":
                time.sleep(STEP_DELAY)
            elif RENDER_MODE == "video":
                frames.append(raw_env.render())

            step += 1

        success = int(bool(info.get("is_success", False)))
        in_box = int(bool(info.get("in_box", False)))
        place_dist = float(info.get("place_dist", np.nan))

        successes += success
        returns.append(ep_ret)

        metrics_data.append({
            "episode": ep + 1,
            "return": ep_ret,
            "success": success,
            "in_box": in_box,
            "place_distance": place_dist
        })

        print(f"  ep {ep}: return={ep_ret:7.2f} "
              f"place_dist={place_dist:.3f} "
              f"in_box={bool(in_box)} "
              f"success={bool(success)}")

        if save_video and frames:
            out_path = os.path.join(VIDEO_DIR, f"{label}_ep{ep}.mp4")
            saved = _save_video(frames, out_path, VIDEO_FPS)
            print(f"    saved {saved}")

    return float(np.mean(returns)), float(np.std(returns)), successes, trajectory_data, metrics_data


def main():
    os.makedirs("results/trajectories", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    paths = expand_paths(MODEL_PATHS)

    if RENDER_MODE == "video":
        os.makedirs(VIDEO_DIR, exist_ok=True)

    results = []

    for i, path in enumerate(paths):
        if not os.path.exists(path):
            print(f"\n[skip] missing: {path}")
            continue

        label = os.path.splitext(os.path.basename(path))[0]

        print(f"\n=== [{i + 1}/{len(paths)}] {path} ===")

        model = SAC.load(path)

        env, raw_env, is_vec, action_mode, normalized_actions, norm_path = make_eval_env(path, model)

        mean_r, std_r, n_succ, trajectory_data, metrics_data = evaluate_one(
            env, raw_env, is_vec, model, label,
            save_video=(RENDER_MODE == "video")
        )

        env.close()

        pd.DataFrame(trajectory_data).to_csv(
            "results/trajectories/SAC_trajectory.csv",
            index=False
        )

        pd.DataFrame(metrics_data).to_csv(
            "results/metrics/SAC_metrics.csv",
            index=False
        )

        results.append((label, path, mean_r, std_r, n_succ))

        print("Saved: results/trajectories/SAC_trajectory.csv")
        print("Saved: results/metrics/SAC_metrics.csv")
        print(f"  -> mean return {mean_r:.2f} ± {std_r:.2f}, success {n_succ}/{EPISODES}")

    print("\n" + "=" * 72)
    print(f"{'model':<30}  {'mean':>10}  {'std':>8}  {'success':>10}")
    print("-" * 72)

    for label, _, mean_r, std_r, n_succ in results:
        print(f"{label:<30}  {mean_r:>10.2f}  {std_r:>8.2f}  {n_succ}/{EPISODES:<8}")


if __name__ == "__main__":
    main()