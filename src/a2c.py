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


class A2CNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super(A2CNet, self).__init__()

        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        nn.init.xavier_uniform_(self.policy[-1].weight, gain=0.01)  # type: ignore

    def forward(self, x):
        body_out = self.body(x)
        return self.policy(body_out), self.value(body_out)


def fill_buffer(
    a2c_net: nn.Module,
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

            action_logits, _ = a2c_net(state_tensor)

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
    a2c_net: nn.Module,
    a2c_optimizer: optim.Optimizer,
    buffer: PolicyTrajectoryBuffer,
    mean_baseline: bool = True,
    entropy_beta: float = 1e-3,
    clip_grad: float = 10,
):
    a2c_net.train()
    losses = []
    entropies = []
    for batch in buffer.get_batches(mean_baseline):
        a2c_optimizer.zero_grad()
        (
            state_batch,
            action_batch,
            reward_batch,
        ) = batch

        logits_v, value_v = a2c_net(state_batch)

        # Value loss
        loss_value_v = F.mse_loss(value_v.squeeze(-1), reward_batch)

        # Policy loss
        log_prob_v = F.log_softmax(logits_v, dim=1)
        adv_v = reward_batch - value_v.detach()
        log_prob_actions_v = adv_v * log_prob_v[range(len(state_batch)), action_batch]
        loss_policy_v = -log_prob_actions_v.mean()

        # Entropy loss
        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = entropy_beta * entropy_v
        loss_policy_v = loss_policy_v - entropy_loss_v

        # Policy backward
        loss_policy_v.backward(retain_graph=True)

        # Value backward
        loss_v = loss_value_v - entropy_loss_v
        loss_v.backward()

        if clip_grad > 0:
            nn.utils.clip_grad_norm_(a2c_net.parameters(), clip_grad)

        a2c_optimizer.step()

        losses.append(loss_v.item() + loss_policy_v.item())
        entropies.append(entropy_v.item())

    return losses, entropies


def evaluate(
    a2c_net: nn.Module,
    env: Environment,
    agent: PolicyAgent,
    device: torch.device,
    random: bool = False,
) -> tuple[str, float]:
    a2c_net.eval()
    max_steps = env.settings.max_steps
    regex_actions = []
    total_reward = 0

    state = env.reset()
    with torch.no_grad():
        for _ in range(len(env)):
            regex_actions = []
            for _ in range(max_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                action_logits, _ = a2c_net(state_tensor)

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
    a2c_net: nn.Module,
    a2c_optimizer: optim.Optimizer,
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
            a2c_net, agent, buffer, env, episodes, epoch=i, device=device
        )
        rewards.extend(train_rewards)

        losses, entropies = train(
            a2c_net, a2c_optimizer, buffer, mean_baseline, entropy_beta, clip_grad
        )

        print(
            f"Epoch {i: >3}/{epochs}:"
            f"\tReward: {np.mean(train_rewards):.1f}"
            f"\tLoss: {np.mean(losses):.3f}"
            + (f"\tEntropy: {np.mean(entropies):.3f}" if verbose_entropy else "")
        )

        if verbose_eval and ((i % eval_period == 0) or (eval_period == (epochs + 1))):
            built_regex, total_reward = evaluate(a2c_net, env, agent, device=device)

            print(f"\nEVALUATION\nRegex: {built_regex}\nTotal reward: {total_reward}\n")

    return rewards
