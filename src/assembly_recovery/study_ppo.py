"""Feedforward study PPO with separate actor/critic and absorbing-slot masks.

This is a project PPO implementation, not the upstream recurrent rl_games
reference. Every future compute-matched study arm must use this same config.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from assembly_recovery.tensor_jobs import finite_job_gae


class RunningNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("variance", torch.ones(size))
        self.register_buffer("count", torch.tensor(0.0001))

    @torch.no_grad()
    def update(self, x, valid):
        weight = valid.to(x.dtype).unsqueeze(-1)
        clean = torch.where(valid.unsqueeze(-1), x, 0.)
        n = weight.sum()
        batch_mean = clean.sum(0) / n.clamp_min(1)
        batch_var = (torch.where(valid.unsqueeze(-1), clean - batch_mean, 0.).square()).sum(0) / n.clamp_min(1)
        total = self.count + n
        delta = batch_mean - self.mean
        variance = (self.variance * self.count + batch_var * n + delta.square() * self.count * n / total) / total
        self.mean.add_(delta * n / total)
        self.variance.copy_(variance)
        self.count.copy_(total)

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.variance + 1e-5)).clamp(-5, 5)


def mlp(size, widths):
    layers = []
    for width in widths:
        layers.extend((nn.Linear(size, width), nn.ELU()))
        size = width
    return nn.Sequential(*layers)


class StudyPolicy(nn.Module):
    def __init__(self, actor_size, critic_size, action_size=7, widths=(512, 128, 64)):
        super().__init__()
        self.spec = {"actor_size": actor_size, "critic_size": critic_size, "action_size": action_size,
                     "widths": list(widths), "recurrent": False, "implementation": "project_feedforward_ppo_v1"}
        self.actor_norm = RunningNorm(actor_size)
        self.critic_norm = RunningNorm(critic_size)
        self.actor = mlp(actor_size, widths)
        self.critic = mlp(critic_size, widths)
        self.mean_head = nn.Linear(widths[-1], action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))
        self.value_head = nn.Linear(widths[-1], 1)

    def normalize(self, observation):
        return {"policy": self.actor_norm(observation["policy"]),
                "critic": self.critic_norm(observation["critic"])}

    def distribution(self, actor_obs):
        mean = self.mean_head(self.actor(actor_obs))
        return Normal(mean, self.log_std.clamp(-5, 2).exp().expand_as(mean), validate_args=False)

    def value(self, critic_obs):
        return self.value_head(self.critic(critic_obs)).squeeze(-1)

    def act(self, observation, deterministic=False):
        normalized = self.normalize(observation)
        distribution = self.distribution(normalized["policy"])
        action = distribution.mean if deterministic else distribution.sample()
        return action, distribution.log_prob(action).sum(-1), self.value(normalized["critic"]), normalized


def ppo_update(model, optimizer, rollout, *, gamma=0.995, tau=0.95, mini_epochs=4, minibatch=512):
    """Update only real job transitions. No value bootstrap at true terminals.

    Normalization statistics update only after the optimization using the exact
    frozen normalized inputs stored during collection. Absorbing observations,
    rewards, actions and values do not enter any loss or running statistic.
    """
    adv, returns = finite_job_gae(rollout["reward"], rollout["value"], rollout["next_value"],
                                 rollout["done"], rollout["valid"], gamma, tau)
    mask = rollout["valid"].flatten()
    indices = mask.nonzero().flatten()  # permitted once at PPO boundary
    if len(indices) == 0:
        return {"updated": False, "active_samples": 0}
    def rows(name):
        x = rollout[name]
        return x.reshape((-1,) + x.shape[2:])[indices]
    actor, critic, action = rows("actor"), rows("critic"), rows("action")
    old_log, old_value = rows("log_prob"), rows("value")
    target = returns.flatten()[indices]
    advantage = adv.flatten()[indices]
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    losses, grad_norms = [], []
    for _ in range(mini_epochs):
        order = torch.randperm(len(indices), device=indices.device)
        for start in range(0, len(indices), minibatch):
            selection = order[start:start + minibatch]
            dist = model.distribution(actor[selection])
            log = dist.log_prob(action[selection]).sum(-1)
            ratio = (log - old_log[selection]).exp()
            policy_loss = -torch.minimum(ratio * advantage[selection], ratio.clamp(0.8, 1.2) * advantage[selection]).mean()
            value = model.value(critic[selection])
            clipped = old_value[selection] + (value - old_value[selection]).clamp(-0.2, 0.2)
            value_loss = torch.maximum((value - target[selection]).square(), (clipped - target[selection]).square()).mean()
            bounds = torch.relu(dist.mean.abs() - 1.1).square().sum(-1).mean()
            loss = policy_loss + value_loss + 0.0001 * bounds
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
            grad_norms.append(norm.detach())
    model.actor_norm.update(rows("raw_actor"), torch.ones(len(indices), dtype=torch.bool, device=indices.device))
    model.critic_norm.update(rows("raw_critic"), torch.ones(len(indices), dtype=torch.bool, device=indices.device))
    return {"updated": True, "active_samples": len(indices), "optimizer_steps": len(losses),
            "mean_loss": float(torch.stack(losses).mean()), "max_gradient_norm_before_clip": float(torch.stack(grad_norms).max())}
