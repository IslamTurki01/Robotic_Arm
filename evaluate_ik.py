import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ik_pickplace as ik


EPISODES = 10
MAX_STEPS = 600
SEED_OFFSET = 0


def main():
    ik.RENDER_MODE = "none"

    env = ik.PickPlaceEnv(render_mode=None, max_steps=MAX_STEPS)

    env.reset(seed=0)
    target_R = ik.compute_home_R(env)

    returns = []
    successes = []
    place_distances = []
    trajectory_data = []

    for ep in range(EPISODES):
        print(f"\n=== Evaluation episode {ep + 1} ===")

        ret, info, frames, episode_trajectory = ik.run_episode(
        env,
        seed=SEED_OFFSET + ep,
        target_R=target_R,
        save_frames=False
    )

        success = int(bool(info.get("is_success", False)))
        place_dist = float(info.get("place_dist", np.nan))

        returns.append(ret)
        successes.append(success)
        place_distances.append(place_dist)

        if episode_trajectory is not None:
            for row in episode_trajectory:
                trajectory_data.append(row)
        print(f"Return: {ret:.2f}")
        print(f"Success: {success}")
        print(f"Place distance: {place_dist:.4f}")

    env.close()

    os.makedirs("results/trajectories", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    if len(trajectory_data) == 0:
        trajectory_df = pd.DataFrame({
            "episode": range(1, EPISODES + 1),
            "step": [0] * EPISODES,
            "actual_x": [np.nan] * EPISODES,
            "actual_y": [np.nan] * EPISODES,
            "actual_z": [np.nan] * EPISODES,
            "reward": returns,
            "is_success": successes,
            "place_dist": place_distances,
            "inference_time": [np.nan] * EPISODES
        })
    else:
        trajectory_df = pd.DataFrame(trajectory_data)

    trajectory_df.to_csv("results/trajectories/IK_trajectory.csv", index=False)

    metrics_df = pd.DataFrame({
        "episode": range(1, EPISODES + 1),
        "return": returns,
        "success": successes,
        "place_distance": place_distances
    })
    metrics_df.to_csv("results/metrics/IK_metrics.csv", index=False)

    success_rate = np.mean(successes) * 100
    mean_place_dist = np.nanmean(place_distances)
    max_place_dist = np.nanmax(place_distances)
    mean_return = np.mean(returns)

    print("\n========== IK Evaluation Results ==========")
    print(f"Success rate / Accuracy: {success_rate:.2f}%")
    print(f"Successful episodes: {sum(successes)} / {EPISODES}")
    print(f"Mean place distance: {mean_place_dist:.4f}")
    print(f"Max place distance: {max_place_dist:.4f}")
    print(f"Mean return: {mean_return:.2f}")

    print("\nSaved:")
    print("results/trajectories/IK_trajectory.csv")
    print("results/metrics/IK_metrics.csv")

    episodes = range(1, EPISODES + 1)

    plt.figure()
    plt.bar(["IK Model"], [success_rate])
    plt.ylabel("Accuracy / Success Rate (%)")
    plt.ylim(0, 100)
    plt.title(f"IK Standard Model Accuracy = {success_rate:.1f}%")
    plt.grid(axis="y")
    plt.savefig("ik_accuracy_plot.png", dpi=200)

    plt.figure()
    plt.plot(episodes, place_distances, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Final Place Distance")
    plt.title("IK Standard Model Final Placement Error")
    plt.grid(True)
    plt.savefig("ik_place_distance_plot.png", dpi=200)

    plt.figure()
    plt.plot(episodes, returns, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("IK Standard Model Return per Episode")
    plt.grid(True)
    plt.savefig("ik_return_plot.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()