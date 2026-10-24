import os
import time
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pickplace_env import PickPlaceEnv


# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "./runs/ppo_pickplace_fast/final_model.zip"
VEC_PATH   = "./runs/ppo_pickplace_fast/vecnormalize.pkl"

RENDER_MODE = "human"      # use "human" to watch, or None for no viewer
EPISODES = 5
MAX_STEPS = 150
ACTION_MODE = "cartesian"
DETERMINISTIC = True
STEP_DELAY = 0.02
# ============================================================


def make_env():
    env = PickPlaceEnv(
        max_steps=MAX_STEPS,
        action_mode=ACTION_MODE,
        render_mode=RENDER_MODE
    )
    return env


env = DummyVecEnv([make_env])

env = VecNormalize.load(VEC_PATH, env)
env.training = False
env.norm_reward = False

model = PPO.load(MODEL_PATH, env=env)


os.makedirs("results/trajectories", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

all_rewards = []
successes = 0
trajectory_data = []
metrics_data = []


for episode in range(EPISODES):
    obs = env.reset()
    done = False
    total_reward = 0.0
    last_info = {}
    step = 0

    while not done:
        start_time = time.perf_counter()

        action, _ = model.predict(obs, deterministic=DETERMINISTIC)

        inference_time = time.perf_counter() - start_time

        obs, reward, done, info = env.step(action)

        total_reward += float(reward[0])
        last_info = info[0]

        # Get real MuJoCo environment from VecNormalize/DummyVecEnv
        raw_env = env.venv.envs[0]

        # Save gripper/pinch position as trajectory
        actual_pos = raw_env.data.site_xpos[raw_env.pinch_site_id].copy()

        trajectory_data.append({
            "episode": episode + 1,
            "step": step,
            "actual_x": actual_pos[0],
            "actual_y": actual_pos[1],
            "actual_z": actual_pos[2],
            "reward": float(reward[0]),
            "is_success": int(bool(last_info.get("is_success", False))),
            "place_dist": float(last_info.get("place_dist", np.nan)),
            "inference_time": inference_time
        })

        env.render()
        time.sleep(STEP_DELAY)

        step += 1

    success = bool(last_info.get("is_success", False))
    in_box = bool(last_info.get("in_box", False))
    place_dist = float(last_info.get("place_dist", -1))

    successes += int(success)
    all_rewards.append(total_reward)

    metrics_data.append({
        "episode": episode + 1,
        "return": total_reward,
        "success": int(success),
        "in_box": int(in_box),
        "place_distance": place_dist
    })

    print(
        f"Episode {episode + 1}: "
        f"reward={total_reward:.2f}, "
        f"place_dist={place_dist:.3f}, "
        f"in_box={in_box}, "
        f"success={success}"
    )


pd.DataFrame(trajectory_data).to_csv(
    "results/trajectories/PPO_trajectory.csv",
    index=False
)

pd.DataFrame(metrics_data).to_csv(
    "results/metrics/PPO_metrics.csv",
    index=False
)


print("\n================ PPO Evaluation Summary ================")
print(f"Episodes: {EPISODES}")
print(f"Mean reward: {np.mean(all_rewards):.2f}")
print(f"Std reward: {np.std(all_rewards):.2f}")
print(f"Success rate: {successes}/{EPISODES}")
print("Saved: results/trajectories/PPO_trajectory.csv")
print("Saved: results/metrics/PPO_metrics.csv")
print("=========================================================")

env.close()