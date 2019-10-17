import numpy as np
import torch
from torch import optim

from utils import run_episode, sample_greedy_return


def compute_reinforce_loss_no_baseline(episode, discount_factor):
    discounted_return_list = []
    log_p_list = []
    G = 0

    for s, a, log_p, s_next, reward in reversed(episode):
        G = reward + discount_factor * G

        discounted_return_list.append(G)
        log_p_list.append(log_p)

    log_p_tensor = torch.stack(log_p_list)
    discounted_return_tensor = torch.FloatTensor(discounted_return_list)

    loss = - torch.sum(log_p_tensor * discounted_return_tensor)

    return loss


def run_episodes_no_baseline(model, env, num_episodes, discount_factor, learn_rate, init_temp, stochasticity):
    optimizer = optim.Adam(model.parameters(), learn_rate)
    episode_durations = []
    losses = []

    for i in range(num_episodes):
        optimizer.zero_grad()

        episode = run_episode(env, model, i, init_temp, stochasticity)
        loss = compute_reinforce_loss_no_baseline(episode, discount_factor)

        loss.backward()
        optimizer.step()

        losses.append(loss.detach().numpy())
        episode_durations.append(len(episode))

        del episode

    return np.asanyarray(episode_durations), np.asanyarray(losses)


def compute_reinforce_loss_with_learned_baseline(value_model, episode, discount_factor, env):
    discounted_return_list = []
    log_p_list = []
    G = 0

    for s, a, log_p, s_next, reward in reversed(episode):
        G = reward + discount_factor * G

        # state = np.unravel_index(s, env.shape)
        baseline = value_model(torch.FloatTensor(s))

        discounted_return_list.append(G - baseline)
        log_p_list.append(log_p)

    log_p_tensor = torch.stack(log_p_list)
    discounted_return_tensor = torch.FloatTensor(discounted_return_list)

    loss = - torch.sum(log_p_tensor * discounted_return_tensor)

    return loss


def compute_value_loss(value_model, episode, discount_factor, env):
    returns = []
    value_estimates = []
    G = 0

    for s, a, log_p, s_next, reward in reversed(episode):
        G = reward + discount_factor * G
        returns.append(G)

        # state = np.unravel_index(s, env.shape)
        value_estimates.append(value_model(torch.FloatTensor(s)))

    value_estimates_tensor = torch.stack(value_estimates)
    returns_tensor = torch.FloatTensor(returns)

    loss = torch.sum(torch.abs(returns_tensor - value_estimates_tensor))

    return loss


def run_episodes_with_learned_baseline(policy_model, value_model, env, num_episodes, discount_factor,
                                       learn_rate_policy, learn_rate_value, init_temp, stochasticity):
    policy_optimizer = optim.Adam(policy_model.parameters(), learn_rate_policy)
    value_optimizer = optim.Adam(value_model.parameters(), learn_rate_value)

    episode_durations = []
    value_losses = []
    reinforce_losses = []

    for i in range(num_episodes):
        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()

        episode = run_episode(env, policy_model, i, init_temp, stochasticity)
        reinforce_loss = compute_reinforce_loss_with_learned_baseline(value_model, episode, discount_factor, env)
        value_loss = compute_value_loss(value_model, episode, discount_factor, env)

        reinforce_loss.backward()
        policy_optimizer.step()

        value_loss.backward()
        value_optimizer.step()

        episode_durations.append(len(episode))
        reinforce_losses.append(reinforce_loss.detach().numpy())
        value_losses.append(value_loss.detach().numpy())

        del episode

    return np.asanyarray(episode_durations), np.asanyarray(reinforce_losses), np.asanyarray(value_losses)


def compute_reinforce_loss_with_SC_baseline(model, episode, discount_factor, env):
    discounted_return_list = []
    log_p_list = []
    G = 0

    for s, a, log_p, s_next, reward in reversed(episode):
        G = reward + discount_factor * G

        baseline = sample_greedy_return(model, env, discount_factor, s)

        discounted_return_list.append(G - baseline)
        log_p_list.append(log_p)

    log_p_tensor = torch.stack(log_p_list)
    discounted_return_tensor = torch.FloatTensor(discounted_return_list)

    loss = - torch.sum(log_p_tensor * discounted_return_tensor)

    return loss


def run_episodes_with_SC_baseline(model, env, num_episodes, discount_factor, learn_rate, init_temp, stochasticity):
    optimizer = optim.Adam(model.parameters(), learn_rate)

    episode_durations = []
    policy_losses = []

    for i in range(num_episodes):
        optimizer.zero_grad()

        episode = run_episode(env, model, i, init_temp, stochasticity)
        loss = compute_reinforce_loss_with_SC_baseline(model, episode, discount_factor, env)

        loss.backward()
        optimizer.step()

        episode_durations.append(len(episode))
        policy_losses.append(loss.detach().numpy())

        del episode

    return np.asanyarray(episode_durations), np.asanyarray(policy_losses)
