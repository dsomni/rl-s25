from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def plot_rewards(rewards: list[float], window: int = 100, title: Optional[str] = None):
    rewards_np = np.array(rewards)
    smoothed = gaussian_filter1d(rewards_np, sigma=window)

    plt.figure(figsize=(12, 7))

    # Raw rewards
    plt.plot(rewards_np, color="gray", alpha=0.3, label="Raw rewards")
    plt.plot(smoothed, linewidth=2.5, label="Smoothed rewards")

    # Max environment reward
    max_reward = np.max(rewards_np)
    plt.axhline(max_reward, color="red", linestyle="--", alpha=0.7, label="Max reward")

    plt.title(title or "Training reward evolution")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
