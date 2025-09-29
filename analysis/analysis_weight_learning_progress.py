import numpy as np
import matplotlib.pyplot as plt
import re
import sys

# Default log file path
DEFAULT_LOG_FILE = "../logs/2025-05-30-10-49-18-online-tamer-cumulative-reward.log-zhihan-pos-neg"

# Path to your log file
log_file = "../logs/2025-05-30-15-33-19-online-tamer-noisy-cumulative-reward.log"


# not noisy
log_file = "../logs/2025-05-30-10-49-18-online-tamer-cumulative-reward.log-zhihan-pos-neg"

# just noisy, 60% w1
# log_file = "../logs/2025-05-30-15-33-19-online-tamer-noisy-cumulative-reward.log"

# noisy, 60%, w1, 0.99 lr decay
# log_file = "../logs/2025-05-30-19-10-37-online-tamer-noisy-cumulative-reward.log"

# noisy, 60%, w2, 0.99 lr decay
# log_file = "../logs/2025-05-30-19-40-00-online-tamer-noisy-cumulative-reward.log"

# noisy, 60%, w2, 0.995 lr decay x5 x0.3
# log_file = "../logs/2025-05-30-21-06-47-online-tamer-noisy-cumulative-reward.log"


def main(log_file_path=None, feature_version="v2"):
    if log_file_path is None:
        log_file_path = DEFAULT_LOG_FILE

    # Initialize containers
    steps = []
    weights = []
    rewards = []
    variances = []

    # Regex patterns
    step_pattern = re.compile(r"step (\d+), weight: \[([^\]]+)\], cumulative reward per episode: ([\-\d.]+).*?variance: ([\d.]+)")

    # Parse the file
    with open(log_file_path, 'r') as f:
        for line in f:
            match = step_pattern.search(line)
            if match:
                step = int(match.group(1))
                weight_str = match.group(2)
                reward = float(match.group(3))
                variance = float(match.group(4))

                # Convert weight string to list of floats
                weight = list(map(float, weight_str.strip().split()))

                # Append to lists
                steps.append(step)
                weights.append(weight)
                rewards.append(reward)
                variances.append(variance)

    # Convert weights to numpy array and calculate stddev
    weights = np.array(weights)
    stddev = np.sqrt(variances)

    # You can now plot with steps, rewards, stddev, and weights
    print("Parsed {} entries.".format(len(steps)))

    # Oracle values
    oracle_mean = 120.8
    oracle_std = np.sqrt(477.96)


        
    # Descriptive labels
    weight_labels = [
        "Dim 1 (# positive left, lower better)",
        "Dim 2 (# negative left, higher better)",
        "Dim 3 (dist to +, lower than Dim 4 better)",
        "Dim 4 (dist to -, higher than Dim 3 better)",
        "Dim 5 (will hit +, higher better)",
        "Dim 6 (will hit -, lower better)"
    ]
    if feature_version == "v3":
        weight_labels = [
            "Dim 1 (collision_passenger)",
            "Dim 2 (collision_obstacle)",
            "Dim 3 (passenger_proximity_score_delta (1/manhattan distance))",
            "Dim 4 (obstacle_proximity_score_delta (1/manhattan distance))",
            "Dim 5 (-)",
            "Dim 6 (-)"
        ]
    if feature_version == "v4":
        weight_labels = [
            "Dim 1 (-)",
            "Dim 2 (-)",
            "Dim 3 (passenger_proximity_score_delta (1/manhattan distance))",
            "Dim 4 (obstacle_proximity_score_delta (1/manhattan distance))",
            "Dim 5 (-)",
            "Dim 6 (-)"
        ]

    # Create the plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top plot: episodic return with oracle
    axs[0].errorbar(steps, rewards, yerr=stddev, fmt='o-', color='black', capsize=5)
    axs[0].axhline(oracle_mean, color='red', linestyle='--', label='Oracle Avg Return')
    axs[0].fill_between(steps, oracle_mean - oracle_std, oracle_mean + oracle_std,
                        color='red', alpha=0.1, label='Oracle Std Dev Range')
    axs[0].set_title("Episodic Return Over Intervention (With Std Dev and Oracle)")
    axs[0].set_ylabel("Avg Episode Return")
    axs[0].legend()
    axs[0].grid(True)

    # Bottom plot: weight trends
    for i in range(weights.shape[1]):
        axs[1].plot(steps, weights[:, i], marker='o', label=weight_labels[i])
    axs[1].set_title("Weight Dimension Trends Over Intervention")
    axs[1].set_xlabel("# Negative Intervention")
    axs[1].set_ylabel("Weight Value")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    # plt.show()

    # save the plot as a png file
    plt.savefig(f"{log_file_path}.png")
    print(f"saved plot to {log_file_path}.png")

if __name__ == "__main__":
    log_file_path = sys.argv[1] if len(sys.argv) > 1 else None
    feature_version = sys.argv[2] if len(sys.argv) > 2 else "v2"
    main(log_file_path, feature_version)


