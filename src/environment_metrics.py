import re
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from dataset import RegexDataset
from rpn import RegexRPN


@dataclass
class EnvSettings:
    invalid_regex_penalty: float = -100
    empty_regex_penalty: float = -100
    word_penalty: float = -1
    length_penalty: float = -0.1

    accuracy_weight: float = 100
    f1_weight: float = 100
    precision_weight: float = 100
    recall_weight: float = 100

    full_match_bonus: float = 100

    max_steps: int = 10

    normalize_state: bool = True


class Environment:
    _finish_action: str = "FIN"

    def __init__(
        self,
        dataset: RegexDataset,
        settings: EnvSettings = EnvSettings(),
    ):
        self.settings = settings

        self.rpn = RegexRPN()
        self._idx_to_action = {
            (i): action for i, action in enumerate(self.rpn.available_tokens)
        }
        self._idx_to_action[len(self._idx_to_action)] = self._finish_action
        self._action_to_idx = {v: k for k, v in self._idx_to_action.items()}
        self._actions = list(self._action_to_idx.keys())

        self.dataset_length = len(dataset)
        self.dataset = dataset.create_iterator()

        self.empty_state_idx = len(self._actions)
        self.finish_action_idx = self._action_to_idx[self._finish_action]

        # self.reset()

    def reset(self) -> np.ndarray:
        self._step_idx = 0
        self.dataset_text, self.dataset_target, self.dataset_words = next(self.dataset)
        self.state = np.array([self.empty_state_idx] * self.settings.max_steps)
        return self.get_state()

    def _get_reward(self) -> float:
        regex_actions = self.state[: self._step_idx]
        try:
            regex_tokens = [self._idx_to_action[x] for x in regex_actions]
            if regex_tokens[-1] == self._finish_action:  # remove finish action if needed
                regex_tokens = regex_tokens[: len(regex_tokens) - 1]
            regex = self.rpn.to_infix(regex_tokens)

            if len(regex) == 0:
                return self.settings.empty_regex_penalty

            matches = [x.span() for x in re.finditer(regex, self.dataset_text)]
        except BaseException:
            return self.settings.invalid_regex_penalty

        target_mask = self.dataset_target
        pred_mask = np.zeros_like(target_mask)

        # Create prediction mask
        for start, end in matches:
            pred_mask[start:end] = 1

        reverse_target_mask = [1 - x for x in target_mask]
        reverse_pred_mask = [1 - x for x in pred_mask]

        # Calculate metrics
        tp = np.logical_and(pred_mask, target_mask).sum()
        fp = np.logical_and(pred_mask, reverse_target_mask).sum()
        fn = np.logical_and(reverse_pred_mask, target_mask).sum()
        tn = np.logical_and(reverse_pred_mask, reverse_target_mask).sum()

        accuracy = (tp + tn) / len(pred_mask)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        full_match = np.bitwise_xor(pred_mask, target_mask).sum() == 0

        words_difference = abs(len(matches) - self.dataset_words)

        return float(
            (f1 - 1) * self.settings.f1_weight
            + (accuracy - 1) * self.settings.accuracy_weight
            # + precision * self.settings.precision_weight
            # + recall * self.settings.recall_weight
            + full_match * self.settings.full_match_bonus
            + words_difference * self.settings.word_penalty
            + len(regex_tokens) * self.settings.length_penalty
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        self.state[self._step_idx] = action
        self._step_idx += 1
        if action == self.finish_action_idx:
            reward = self._get_reward()
            self.reset()
            return self.get_state(), reward, True

        if self._step_idx >= self.settings.max_steps:
            reward = self._get_reward()
            self.reset()
            return self.get_state(), reward, True

        return self.get_state(), 0, False

    def get_state(self) -> np.ndarray:
        if not self.settings.normalize_state:
            return self.state
        return self.state / self.empty_state_idx

    @cached_property
    def action_space(self) -> int:
        return len(self._actions)

    @cached_property
    def state_space(self) -> int:
        return self.settings.max_steps

    @cached_property
    def actions(self) -> list[str]:
        return self._actions

    def action_to_idx(self, action: str) -> int:
        return self._action_to_idx[action]

    def idx_to_action(self, action_idx: int) -> str:
        return self._idx_to_action[action_idx]

    def __len__(self):
        return self.dataset_length
