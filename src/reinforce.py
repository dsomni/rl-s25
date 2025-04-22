import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from environment_metrics import Environment
from rl import PolicyAgent, PolicyTrajectoryBuffer
from utils import set_seed

ROOT_FOLDER = os.path.join(".", "..")
if ROOT_FOLDER not in sys.path:
    sys.path.insert(0, ROOT_FOLDER)


# from environment import Environment, EnvSettings


class PGN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def fill_buffer(
    pgn: nn.Module,
    agent: PolicyAgent,
    buffer: PolicyTrajectoryBuffer,
    env: Environment,
    episodes: int,
    epoch: int,
    device: torch.device,
):
    buffer.clean()
    state = env.reset()
    done_episodes = 0
    ep_states_buf, ep_rew_act_buf = [], []

    train_rewards = []

    epoch_loop = tqdm(total=episodes, desc=f"Epoch #{epoch}", position=0, disable=True)

    with torch.no_grad():
        while done_episodes < episodes:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)

            action_logits = pgn(state_tensor)

            action = agent.choose_action(action_logits, epoch=epoch)
            next_state, reward, done = env.step(action)

            ep_states_buf.append(state)
            ep_rew_act_buf.append([reward, int(action)])

            state = next_state

            if done:
                buffer.store(
                    np.array(ep_states_buf),
                    np.array(ep_rew_act_buf),
                )

                ep_states_buf, ep_rew_act_buf = [], []

                train_rewards.append(reward)

                done_episodes += 1
                epoch_loop.update(1)

    return train_rewards


def train(
    pgn: nn.Module,
    pgn_optimizer: optim.Optimizer,
    buffer: PolicyTrajectoryBuffer,
    mean_baseline: bool = True,
    entropy_beta: float = 1e-3,
    clip_grad: float = 10,
):
    pgn.train()
    losses = []
    entropies = []
    for batch in buffer.get_batches(mean_baseline):
        pgn_optimizer.zero_grad()
        (
            state_batch,
            action_batch,
            reward_batch,
        ) = batch

        logits_v = pgn(state_batch)

        # Policy loss
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = (
            reward_batch * log_prob_v[range(len(state_batch)), action_batch]
        )
        loss_policy_v = -log_prob_actions_v.mean()

        # Entropy loss
        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = entropy_beta * entropy_v
        loss_policy_v = loss_policy_v - entropy_loss_v

        # Policy backward
        loss_v = loss_policy_v - entropy_loss_v
        loss_v.backward()

        if clip_grad > 0:
            nn.utils.clip_grad_norm_(pgn.parameters(), clip_grad)

        pgn_optimizer.step()

        losses.append(loss_v.item())
        entropies.append(entropy_v.item())

    return losses, entropies


def evaluate(
    pgn: nn.Module,
    env: Environment,
    agent: PolicyAgent,
    device: torch.device,
    random: bool = False,
) -> tuple[str, float]:
    pgn.eval()
    max_steps = env.settings.max_steps
    regex_actions = []
    total_reward = 0

    state = env.reset()
    with torch.no_grad():
        for _ in range(len(env)):
            regex_actions = []
            for _ in range(max_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                action_logits = pgn(state_tensor)

                if random:
                    action = np.random.randint(env.action_space)
                else:
                    action = agent.choose_optimal_action(action_logits)
                regex_actions.append(env.idx_to_action(action))

                next_state, reward, done = env.step(action)

                state = next_state
                if done:
                    total_reward += reward
                    break

    if regex_actions and regex_actions[-1] == env.finish_action:
        regex_actions = regex_actions[:-1]

    try:
        regex = env.rpn.to_infix(regex_actions)
    except BaseException:
        regex = f"Invalid: {regex_actions}"

    return regex, total_reward


def train_eval_loop(
    pgn: nn.Module,
    pgn_optimizer: optim.Optimizer,
    agent: PolicyAgent,
    buffer: PolicyTrajectoryBuffer,
    env: Environment,
    epochs: int,
    episodes: int,
    device: torch.device,
    mean_baseline: bool = True,
    entropy_beta: float = 0.5,
    eval_period: int = 5,
    clip_grad: float = 10,
    verbose_eval: bool = False,
    verbose_entropy: bool = False,
) -> list[float]:
    set_seed()

    rewards = []

    for i in range(1, epochs + 1):
        train_rewards = fill_buffer(
            pgn, agent, buffer, env, episodes, epoch=i, device=device
        )
        rewards.extend(train_rewards)

        losses, entropies = train(
            pgn, pgn_optimizer, buffer, mean_baseline, entropy_beta, clip_grad
        )

        print(
            f"Epoch {i: >3}/{epochs}:"
            f"\tReward: {np.mean(train_rewards):.1f}"
            f"\tLoss: {np.mean(losses):.3f}"
            + (f"\tEntropy: {np.mean(entropies):.3f}" if verbose_entropy else "")
        )

        if verbose_eval and ((i % eval_period == 0) or (eval_period == (epochs + 1))):
            built_regex, total_reward = evaluate(pgn, env, agent, device=device)

            print(f"\nEVALUATION\nRegex: {built_regex}\nTotal reward: {total_reward}\n")

    return rewards
