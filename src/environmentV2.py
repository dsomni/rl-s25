import re
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from dataset import RegexDataset
from rpn import RegexRPN


@dataclass
class EnvSettings:
    invalid_regex_penalty: float = -100000
    word_penalty: float = -10000
    token_penalty: float = -100
    length_penalty: float = -1

    max_steps: int = 10

    normalize_state: bool = True


class EnvironmentV2:
    _finish_action: str = "FIN"

    def __init__(
        self,
        dataset: RegexDataset,
        settings: EnvSettings = EnvSettings(),
        penalty_weights: dict = None
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

        default_weights = {
            'f1': 10.0,
            'precision': 2.0,
            'recall': 2.0,
            'complexity': -0.5,
            'full_match': 50.0,
            'syntax_error': -100.0,
            'partial_progress': 5.0,
            'length_penalty': -0.3
        }
        self.penalty_weights = penalty_weights or default_weights
        self.regex = None

        self.reset()

    def reset(self) -> np.ndarray:
        self._step_idx = 0
        self.dataset_text, self.dataset_target, self.dataset_words = next(self.dataset)
        self.state = np.array([self.empty_state_idx] * self.settings.max_steps)
        self.regex_history = []
        return self.get_state()

    def _get_reward(self) -> float:
        regex_actions = self.state[: self._step_idx]
        try:
            regex_tokens = [self._idx_to_action[x] for x in regex_actions]
            if regex_tokens[-1] == self._finish_action:  # remove finish action if needed
                regex_tokens = regex_tokens[: len(regex_tokens) - 1]
            regex = self.rpn.to_infix(regex_tokens)
            self.regex_history.append(regex)

            array = [x.span() for x in re.finditer(regex, self.dataset_text)]
        except BaseException:
            return self.settings.invalid_regex_penalty

        pred_mask = [0 for _ in range(len(self.dataset_text))]
        target_mask = self.dataset_target

        for it in array:
            for i in range(it[0], it[1]):
                pred_mask[i] = 1
        
        reverse_target_mask = [1 - x for x in target_mask]
        reverse_pred_mask = [1 - x for x in pred_mask]
        
        # Calculate metrics
        tp = np.logical_and(pred_mask, target_mask).sum()
        fp = np.logical_and(pred_mask, reverse_target_mask).sum()
        fn = np.logical_and(reverse_pred_mask, target_mask).sum()
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        self.regex = regex
        # print(regex)
        # print(re.findall(regex, self.dataset_text))
        # print(bit_mask)
        # print("----")

        reward_components = {
            'f1': f1 * self.penalty_weights['f1'],
            'precision': precision * self.penalty_weights['precision'],
            'recall': recall * self.penalty_weights['recall'],
            # 'complexity': len(regex_str) * self.penalty_weights['complexity'],
            'length_penalty': len(regex_actions) * self.penalty_weights['length_penalty']
        }
        
        # Full match bonus
        if f1 >= 0.99:
            reward_components['full_match'] = self.penalty_weights['full_match']
        
        # Partial progress bonus (compare with previous attempts)
        if len(self.regex_history) > 1:
            prev_f1 = self._calculate_metrics(self.regex_history[-2])[0] or 0
            reward_components['partial_progress'] = self.penalty_weights['partial_progress'] * (f1 - prev_f1)
        
        # Total reward calculation
        total_reward = sum(reward_components.values())
        
        # Apply non-linear scaling
        return np.sign(total_reward) * np.log1p(np.abs(total_reward))

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
        return 1 - (self.state / self.empty_state_idx)

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
    
    def get_regexp(self):
        return self.regex
