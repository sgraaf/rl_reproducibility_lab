import random

import numpy as np
import torch
from matplotlib import pyplot as plt


def get_running_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return round(h), round(m), round(s)


def set_seeds(env, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)


def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def plot_episodes_durations_losses(episode_durations, policy_losses, value_losses, title):
    N = len(episode_durations)

    plt.figure()
    for i in range(N):
        plt.plot(smooth(episode_durations[i], 10), label=f'Run {i}')
        plt.title(f'Duration per Episode for {title}')
        plt.xlabel('# Episodes')
        plt.ylabel('# Steps to Termination')
        plt.legend()

    plt.figure()
    for i in range(N):
        plt.plot(smooth(policy_losses[i], 10), label=f'Run {i}')
        plt.title(f'Policy loss per Episode for {title}')
        plt.xlabel('# Episodes')
        plt.ylabel('Loss')
        plt.legend()

    if value_losses:
        plt.figure()
        for i in range(N):
            plt.plot(smooth(value_losses[i], 10), label=f'Run {i}')
            plt.title(f'Value loss per Episode for {title}')
            plt.xlabel('# Episodes')
            plt.ylabel('Loss')
            plt.legend()


def select_action(model, state, epoch, grid_shape, init_temperature=1.1):
    """
    Samples an action according to the probability distribution induced by the model
    Also returns the log_probability
    """
    state = np.unravel_index(state, grid_shape)
    log_p = model(torch.FloatTensor(state))
    
    # now add the temperature for making some exploration.
    decay_exploration_epochs = 50
    temperature = 1
    if epoch < decay_exploration_epochs:
        temperature = init_temperature - (init_temperature-temperature)*(epoch/decay_exploration_epochs)

    probs = log_p.div(temperature)
    probs = torch.exp(probs)
    probs = probs / torch.sum(probs)
    action = torch.multinomial(probs, 1).item()

    return action, log_p[action]


def run_episode(env, model, epoch, grid_shape):
    episode = []
    
    s = env.reset()
    done = False
    step = 0
    max_steps = 50
    
    while not done and step < max_steps:
        
        a, log_p = select_action(model, s, epoch, grid_shape)
        s_next, r, done, _ = env.step(a)
        
        episode.append((s, a, log_p, s_next, r))
        s = s_next
        step += 1
          
    env.close()
    
    return episode


def sample_greedy_return(model, env, discount_factor, grid_shape, s=None):
    
    if s is None:
        s = env.reset()
    else:
        _ = env.reset()
        env.s = s
    
    done = False
    step = 0
    max_steps = 50
    greedy_return = 0
    
    while not done and step < max_steps:
        
        state = np.unravel_index(s, grid_shape)
        log_p = model(torch.FloatTensor(state))
        
        greedy_a =  log_p.max(0)[1].item()
        s, reward, done, _ = env.step(greedy_a)
        
        greedy_return = reward + discount_factor * greedy_return
        
        step += 1

    env.close()
    
    return greedy_return

def compare_baselines_plot(baselines_dict):
    # episode durations
    plt.figure()
    for baseline in baselines_dict:
        mean = np.mean(baselines_dict[baseline]['episode_durations'], axis=0)
        std = np.std(baselines_dict[baseline]['episode_durations'], axis=0)

        plt.plot(smooth(mean, 10), label=baseline, color=baselines_dict[baseline]['color'])
        plt.fill_between(range(len(smooth(mean, 10))), smooth(mean, 10) - smooth(std, 10), smooth(mean, 10) + smooth(std, 10), color=baselines_dict[baseline]['color'], alpha=0.2)

    plt.title('Mean episode duration per Episode')
    plt.xlabel('# Episodes')
    plt.ylabel('Mean episode duration')
    plt.legend()
    
    # policy losses
    plt.figure()
    for baseline in baselines_dict:
        mean = np.mean(baselines_dict[baseline]['policy_loss'], axis=0)
        std = np.std(baselines_dict[baseline]['policy_loss'], axis=0)

        plt.plot(smooth(mean, 10), label=baseline, color=baselines_dict[baseline]['color'])
        plt.fill_between(range(len(smooth(mean, 10))), smooth(mean, 10) - smooth(std, 10), smooth(mean, 10) + smooth(std, 10), color=baselines_dict[baseline]['color'], alpha=0.2)

    plt.title('Mean policy loss per Episode')
    plt.xlabel('# Episodes')
    plt.ylabel('Mean loss')
    plt.legend()