"""Train SAC on the pick-and-place task.

Edit the CONFIG block below, then just run:  python train_pickplace.py
"""

import os
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pickplace_env import PickPlaceEnv
# ============================================================
# CONFIG — FAST SAC VERSION FOR COLAB / LAPTOP
# ============================================================
TIMESTEPS = 50_000
LOGDIR    = "./runs/sac_pickplace_fast"
SEED      = 0

RESUME_FROM = None
VECNORMALIZE_FROM = None

# SAC hyperparameters - reduced for speed
LEARNING_RATE   = 3e-4
BUFFER_SIZE     = 50_000
BATCH_SIZE      = 64
TAU             = 0.005
GAMMA           = 0.99
LEARNING_STARTS = 500
NET_ARCH        = [128,128]

# Environment/training setup - reduced for speed
N_ENVS                = 1
ACTION_MODE           = "cartesian"
NORMALIZE_ACTIONS     = True
MAX_STEPS             = 600
TRAIN_START_LEVEL     = 0
EVAL_CURRICULUM_LEVEL = 1
NEAR_CUBE_RESET_PROB  = 0.60
PREGRASP_RESET_PROB   = 0.90

# Callback cadence
CHECKPOINT_FREQ   = 10_000
EVAL_FREQ         = 20_000
N_EVAL_EPISODES   = 1
# =======================================================================================================================


class CurriculumCallback(BaseCallback):
    """Increase reset randomization as training progresses."""

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.current_level = None

    def _on_step(self) -> bool:
        progress = self.num_timesteps / max(1, self.total_timesteps)
        if progress < 0.30:
            level = 0
            near_prob = NEAR_CUBE_RESET_PROB
            pregrasp_prob = PREGRASP_RESET_PROB
        elif progress < 0.65:
            level = 1
            near_prob = 0.25
            pregrasp_prob = 0.35
        else:
            level = 2
            near_prob = 0.10
            pregrasp_prob = 0.15

        if level != self.current_level:
            self.training_env.env_method("set_curriculum_level", level)
            self.training_env.env_method(
                "set_reset_curriculum",
                near_prob,
                pregrasp_prob,
            )
            self.current_level = level
            print(
                f"[curriculum] level={level}, "
                f"near_cube_reset_prob={near_prob:.2f}, "
                f"pregrasp_reset_prob={pregrasp_prob:.2f}"
            )
        return True


def callback_freq(freq: int) -> int:
    return max(freq // max(1, N_ENVS), 1)


def rescale_to_unit_action(env):
    low = np.full(env.action_space.shape, -1.0, dtype=np.float32)
    high = np.full(env.action_space.shape, 1.0, dtype=np.float32)
    return gym.wrappers.RescaleAction(env, low, high)


def find_resume_vecnormalize_path():
    if VECNORMALIZE_FROM:
        return VECNORMALIZE_FROM

    candidates = []
    if RESUME_FROM:
        model_dir = os.path.dirname(RESUME_FROM)
        model_base = os.path.splitext(os.path.basename(RESUME_FROM))[0]
        if model_base.startswith("sac_"):
            candidates.append(
                os.path.join(
                    model_dir,
                    model_base.replace("sac_", "sac_vecnormalize_", 1) + ".pkl",
                )
            )
        candidates.extend([
            os.path.join(model_dir, "vecnormalize.pkl"),
            os.path.join(os.path.dirname(model_dir), "vecnormalize.pkl"),
        ])

    candidates.append(os.path.join(LOGDIR, "vecnormalize.pkl"))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def make_env(
    seed: int = 0,
    curriculum_level: int = TRAIN_START_LEVEL,
    near_cube_reset_prob: float = NEAR_CUBE_RESET_PROB,
    pregrasp_reset_prob: float = PREGRASP_RESET_PROB,
):
    def _init():
        env = PickPlaceEnv(
            max_steps=MAX_STEPS,
            curriculum_level=curriculum_level,
            action_mode=ACTION_MODE,
            near_cube_reset_prob=near_cube_reset_prob,
            pregrasp_reset_prob=pregrasp_reset_prob,
        )
        if NORMALIZE_ACTIONS:
            env = rescale_to_unit_action(env)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def make_vec_envs():
    train_env = DummyVecEnv([
        make_env(
            SEED + i,
            TRAIN_START_LEVEL,
            NEAR_CUBE_RESET_PROB,
            PREGRASP_RESET_PROB,
        )
        for i in range(N_ENVS)
    ])

    eval_env = DummyVecEnv([
        make_env(SEED + 10_000, EVAL_CURRICULUM_LEVEL, 0.0, 0.0)
    ])

    norm_path = find_resume_vecnormalize_path()
    if RESUME_FROM and norm_path and os.path.exists(norm_path):
        print(f"Loading VecNormalize stats from {norm_path}")
        train_env = VecNormalize.load(norm_path, train_env)
        train_env.training = True
        train_env.norm_reward = False
    elif RESUME_FROM:
        print("[warn] RESUME_FROM set, but no VecNormalize stats were found")
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
    else:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )

    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )
    return train_env, eval_env


def main():
    os.makedirs(LOGDIR, exist_ok=True)

    train_env, eval_env = make_vec_envs()

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Resuming from {RESUME_FROM}")
        model = SAC.load(RESUME_FROM, env=train_env)
    else:
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            tau=TAU,
            gamma=GAMMA,
            train_freq=1,
           gradient_steps= 2,
            learning_starts=LEARNING_STARTS,
            policy_kwargs=dict(net_arch=NET_ARCH),
            tensorboard_log=LOGDIR,
            verbose=1,
            seed=SEED,
            device="cuda"
        )

    callbacks = [
        CurriculumCallback(TIMESTEPS),
        CheckpointCallback(
            save_freq=callback_freq(CHECKPOINT_FREQ),
            save_path=os.path.join(LOGDIR, "ckpts"),
            name_prefix="sac",
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(LOGDIR, "best"),
            log_path=os.path.join(LOGDIR, "eval"),
            eval_freq=callback_freq(EVAL_FREQ),
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
        ),
    ]



    start_train_time = time.perf_counter()

    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callbacks,
        progress_bar=True
    )

    end_train_time = time.perf_counter()
    training_time = end_train_time - start_train_time

    with open(os.path.join(LOGDIR, "training_time.txt"), "w") as f:
        f.write(str(training_time))

    print(f"SAC training time: {training_time:.2f} seconds")
    
    train_env.save(os.path.join(LOGDIR, "vecnormalize.pkl"))
    model.save(os.path.join(LOGDIR, "final_model"))
    print(f"Saved final model to {LOGDIR}/final_model.zip")


if __name__ == "__main__":
    main()
