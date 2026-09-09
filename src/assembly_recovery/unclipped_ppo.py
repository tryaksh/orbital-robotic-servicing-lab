"""Reward-v3 PPO with unclipped value regression; actor clipping remains 0.2.

The controlled critic-fit probe motivates this single optimizer correction.
Architecture, GAE, normalization, gradient clipping and update budget are retained.
"""
import torch

from assembly_recovery.tensor_jobs import finite_job_gae


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
    old_log = rows("log_prob")
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
            value_loss = (value - target[selection]).square().mean()
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
            "value_prediction_mean": float(rows("value").mean()), "value_target_mean": float(target.mean()),
            "value_rmse_before_update": float((rows("value") - target).square().mean().sqrt()),
            "mean_loss": float(torch.stack(losses).mean()), "max_gradient_norm_before_clip": float(torch.stack(grad_norms).max())}
