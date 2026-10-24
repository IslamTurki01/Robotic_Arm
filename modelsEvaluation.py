import os
import time
import numpy as np
import pandas as pd


def compute_rmse(reference, predicted):
    min_len = min(len(reference), len(predicted))
    reference = reference[:min_len]
    predicted = predicted[:min_len]
    return np.sqrt(np.mean((reference - predicted) ** 2))


def compute_mae(reference, predicted):
    min_len = min(len(reference), len(predicted))
    reference = reference[:min_len]
    predicted = predicted[:min_len]
    return np.mean(np.abs(reference - predicted))


def generate_reference_trajectory(start, goal, steps):
    return np.linspace(start, goal, steps)


def read_training_time(path):
    try:
        with open(path, "r") as f:
            return float(f.read())
    except FileNotFoundError:
        return 0.0


def clean_xyz(df):
    cols = ["actual_x", "actual_y", "actual_z"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=cols)
    return df[cols].values.astype(float)


def load_trajectory(path):
    df = pd.read_csv(path)
    return clean_xyz(df)


def load_metrics(path):
    if not os.path.exists(path):
        return np.nan, np.nan, np.nan

    df = pd.read_csv(path)

    success_rate = df["success"].mean() * 100 if "success" in df.columns else np.nan

    if "return" in df.columns:
        mean_return = df["return"].mean()
    elif "total_reward" in df.columns:
        mean_return = df["total_reward"].mean()
    else:
        mean_return = np.nan

    if "place_distance" in df.columns:
        mean_place_distance = df["place_distance"].mean()
    elif "place_dist" in df.columns:
        mean_place_distance = df["place_dist"].mean()
    else:
        mean_place_distance = np.nan

    return success_rate, mean_return, mean_place_distance

def evaluate_model(model_name, trajectory_path, metrics_path, training_time_path, reference_trajectory):
    start_inference = time.perf_counter()

    predicted_trajectory = load_trajectory(trajectory_path)

    end_inference = time.perf_counter()
    inference_time = end_inference - start_inference

    steps = min(len(reference_trajectory), len(predicted_trajectory))
    reference = reference_trajectory[:steps]
    predicted = predicted_trajectory[:steps]

    rmse = compute_rmse(reference, predicted)
    mae = compute_mae(reference, predicted)

    training_time = read_training_time(training_time_path)
    success_rate, mean_return, mean_place_distance = load_metrics(metrics_path)

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "Success Rate (%)": success_rate,
        "Mean Return": mean_return,
        "Mean Place Distance (m)": mean_place_distance,
        "Training Time (sec)": training_time,
        "Inference Time (sec)": inference_time,
        "Trajectory Steps Used": steps
    }


if __name__ == "__main__":

    start_position = np.array([0.0, 0.0, 0.0])
    goal_position = np.array([0.5, 0.3, 0.2])

    reference_trajectory = generate_reference_trajectory(
        start=start_position,
        goal=goal_position,
        steps=600
    )

    models = [
        ("SAC",
         "results/trajectories/SAC_trajectory.csv",
         "results/metrics/SAC_metrics.csv",
         "runs/sac_pickplace_fast/training_time.txt"),

        ("PPO",
         "results/trajectories/PPO_trajectory.csv",
         "results/metrics/PPO_metrics.csv",
         "runs/ppo_pickplace_fast/training_time.txt"),

        ("IK",
         "results/trajectories/IK_trajectory.csv",
         "results/metrics/IK_metrics.csv",
         ""),
        ("PID",
        "results/trajectories/PID_trajectory.csv",
        "results/pid_pickplace/episode_summary.csv",
        ""),

        ("SINDy",
        "results/trajectories/SINDy_trajectory.csv",
        "results/sindy_gd_pickplace/episode_summary.csv",
        "results/sindy_gd_pickplace/training_time.txt"),
    ]

    results = []

    for model_name, trajectory_path, metrics_path, training_time_path in models:
        if not os.path.exists(trajectory_path):
            print(f"Missing trajectory file for {model_name}: {trajectory_path}")
            continue

        results.append(
            evaluate_model(
                model_name,
                trajectory_path,
                metrics_path,
                training_time_path,
                reference_trajectory
            )
        )

    df = pd.DataFrame(results)

    print("\nFinal Model Comparison:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df)

    df.to_csv("model_comparison_results.csv", index=False)

    print("\nSaved as model_comparison_results.csv")