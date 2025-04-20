import re
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from dataset import RegexDataset
from rpn import RegexRPN


@dataclass
class EnvSettings:
    token_penalty: float = -100
    word_penalty: float = -10000
    length_penalty: float = -1

    invalid_regex_penalty: float = -100000

    max_steps: int = 5


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

        self.dataset = dataset.create_iterator()

        self.empty_state_idx = len(self._actions)
        self.finish_action_idx = self._action_to_idx[self._finish_action]

        self.reset()

    def reset(self) -> np.ndarray:
        self._step_idx = 0
        self.dataset_text, self.dataset_target, self.dataset_words = next(self.dataset)
        self.state = np.array([self.empty_state_idx] * self.settings.max_steps)
        return self.state

    def _get_reward(self) -> float:
        regex_actions = self.state[: self._step_idx]
        try:
            regex_tokens = [self._idx_to_action[x] for x in regex_actions]
            if regex_tokens[-1] == self._finish_action:  # remove finish action if needed
                regex_tokens = regex_tokens[: len(regex_tokens) - 1]
            regex = self.rpn.to_infix(regex_tokens)

            array = [x.span() for x in re.finditer(regex, self.dataset_text)]
        except RuntimeError:
            return self.settings.invalid_regex_penalty

        bit_mask = [0 for _ in range(len(self.dataset_text))]

        for it in array:
            for i in range(it[0], it[1]):
                bit_mask[i] = 1

        tokens_difference = sum(np.bitwise_xor(bit_mask, self.dataset_target))
        words_difference = abs(len(array) - self.dataset_words)
        return (
            tokens_difference * self.settings.token_penalty
            + words_difference * self.settings.word_penalty
            + len(regex_tokens) * self.settings.length_penalty
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        self.state[self._step_idx] = action
        self._step_idx += 1
        if action == self.finish_action_idx:
            reward = self._get_reward()
            state = self.state
            self.reset()
            return state, reward, True

        if self._step_idx > self.settings.max_steps:
            self.reset()

        return self.state, self._get_reward(), False

    @cached_property
    def action_space(self) -> int:
        return len(self._actions)

    @cached_property
    def actions(self) -> list[str]:
        return self._actions
