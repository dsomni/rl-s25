from itertools import accumulate

import numpy as np
import torch
import torch.nn.functional as F


def calculate_qvals(
    rewards: list[float] | np.ndarray, gamma: float = 1.0, reward_steps: int = 0
) -> np.ndarray:
    rw_steps = reward_steps if reward_steps != 0 else len(rewards)

    return np.array(
        [
            list(
                accumulate(
                    reversed(rewards[i : i + rw_steps]), lambda x, y: gamma * x + y
                )
            )[-1]
            for i in range(len(rewards))
        ]
    )


class PolicyAgent:
    def __init__(self, temperature_coefficient: float = 10.0):
        self.temperature_coefficient = temperature_coefficient

    def choose_action(self, action_logits: torch.Tensor, epoch: int):
        temperature = (
            1 / epoch * torch.max(torch.abs(action_logits)) * self.temperature_coefficient
            if self.temperature_coefficient > 0
            else 1
        )

        return np.random.choice(
            range(len(action_logits)),
            size=1,
            p=F.softmax(action_logits / temperature, dim=0).numpy(),
        )[0]

    def choose_optimal_action(self, action_logits: torch.Tensor) -> int:
        return int(np.argmax(F.softmax(action_logits, dim=0).cpu()).item())


class PolicyTrajectoryBuffer:
    """
    Buffer class to store the experience from a unique policy
    """

    def _batch(self, iterable):
        ln = len(iterable)
        for ndx in range(0, ln, self.batch_size):
            yield iterable[ndx : min(ndx + self.batch_size, ln)]

    def __init__(self, device: torch.device, batch_size: int = 64):
        self.device = device
        self.batch_size = batch_size
        self.clean()

    def clean(self):
        self.states = []
        self.actions = []
        self.discounted_rewards = []

    def store(
        self,
        states_trajectory: np.ndarray,
        trajectory: np.ndarray,
    ):
        """
        Add trajectory values to the buffers and compute the advantage and reward to go

        Parameters:
        -----------
        states_trajectory:  list that contains states
        trajectory: list where each element is a list that contains: reward, action
        """
        assert len(states_trajectory) == len(trajectory)

        if len(states_trajectory) > 0:
            self.states.extend(states_trajectory)
            self.actions.extend(trajectory[:, 1])

            self.discounted_rewards.extend(calculate_qvals(trajectory[:, 0]))

    def get_batches(self, mean_baseline: bool):
        mean_rewards = np.mean(self.discounted_rewards) if mean_baseline else 0

        for states_batch, actions_batch, discounted_rewards_batch in zip(
            self._batch(self.states),
            self._batch(self.actions),
            self._batch(self.discounted_rewards),
        ):
            yield (
                torch.tensor(states_batch, dtype=torch.float32, device=self.device),
                torch.tensor(actions_batch, dtype=torch.long, device=self.device),
                torch.tensor(
                    np.array(discounted_rewards_batch) - mean_rewards,
                    dtype=torch.float,
                    device=self.device,
                ),
            )

    def __len__(self):
        return len(self.states)
