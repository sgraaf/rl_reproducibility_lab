from time import time
import pickle as pkl

import os 
import gym
import numpy as np

from envs.gridworld import GridworldEnv
from models import PolicyNetwork, ValueNetwork
from utils import get_running_time, set_seeds
from run_loss_functions import run_episodes_no_baseline, run_episodes_with_learned_baseline, run_episodes_with_SC_baseline

stochasticity = 0.0  # <---------- change this!
n_runs = 5 # <---------- change this after you made sure it works (you created the nec. folders for the results)!
n_episodes = 750

def run_learned_baseline(stochasticity, n_runs, n_episodes):
    # learned baseline
    dir_path = os.path.dirname(os.path.realpath(__file__))
    best_settings_file = dir_path+f'/cart_pole_parameter_search/s{stochasticity}_learned_baseline_best_settings.pkl'
    eval_file = f'cart_evals/s{stochasticity}_learned_baseline.pkl'

    with open(best_settings_file, 'rb') as pickle_file:
        best_settings = pkl.load(pickle_file)
    discount_factor = best_settings['discount_factor'] 
    learn_rate_policy = best_settings['learn_rate_policy']
    learn_rate_value = best_settings['learn_rate_value']
    hidden_dim_policy = best_settings['hidden_dim_policy']
    hidden_dim_value = best_settings['hidden_dim_value']
    init_temp = best_settings['init_temp']

    st = time()

    # change this for learned baseline
    print(f'Run settings: baseline=run_episodes_with_learned_baseline, discount_factor={discount_factor}, learn_rate_policy={learn_rate_policy}, learn_rate_value={learn_rate_value}, hidden_dim_policy={hidden_dim_policy}, hidden_dim_value={hidden_dim_value}, init_temp={init_temp}')

    # initialize the environment
    env = gym.make('CartPole-v1')  

    episode_durations_list = []
    reinforce_loss_list = []
    value_loss_list = []


    for i in range(n_runs):
        start_time = time()

        policy_model = PolicyNetwork(input_dim=4, hidden_dim=hidden_dim_policy, output_dim=2)  # change input_ and output_dim for gridworld env
        value_model = ValueNetwork(input_dim=4, hidden_dim=hidden_dim_value)  # change input_dim for gridworld env
        seed = 40 + i
        set_seeds(env, seed)

        episode_durations, reinforce_loss, value_loss = run_episodes_with_learned_baseline( 
            policy_model,
            value_model,
            env,
            n_episodes,
            discount_factor,
            learn_rate_policy,
            learn_rate_value,
            init_temp,
            stochasticity
        )

        episode_durations_list.append(episode_durations)
        reinforce_loss_list.append(reinforce_loss)
        value_loss_list.append(value_loss)

        del policy_model
        del value_model

        end_time = time()
        h, m, s = get_running_time(end_time - start_time)

        print(f'Done with run {i+1}/{n_runs} in {f"{h} hours, " if h else ""}{f"{m} minutes and " if m else ""}{s} seconds')

    env.close()

    et = time()
    h, m, s = get_running_time(et - st)

    evals = {}
    evals['episode_durations'] = episode_durations_list
    evals['reinforce_loss'] = reinforce_loss_list
    evals['value_loss'] = value_loss_list

    pkl.dump(evals, open(eval_file, 'wb'))

    print(f'Done with run in {f"{h} hours, " if h else ""}{f"{m} minutes and " if m else ""}{s} seconds')


def run_selfcritic_baseline(stochasticity, n_runs, n_episodes):         
    # self-critic baseline
    dir_path = os.path.dirname(os.path.realpath(__file__))
    best_settings_file = dir_path+f'/cart_pole_parameter_search/s{stochasticity}_SC_baseline_best_settings.pkl'
    eval_file = f'cart_evals/s{stochasticity}_SC_baseline.pkl'

    with open(best_settings_file, 'rb') as pickle_file:
        best_settings = pkl.load(pickle_file)
    discount_factor = best_settings['discount_factor'] 
    learn_rate = best_settings['learn_rate']
    hidden_dim = best_settings['hidden_dim']
    init_temp = best_settings['init_temp']
                    
    st = time()

    # change this for learned baseline
    print(f'Run settings: baseline=run_episodes_with_SC_baseline, discount_factor={discount_factor}, learn_rate={learn_rate}, hidden_dim={hidden_dim}, init_temp={init_temp}')

    # initialize the environment
    env = gym.make('CartPole-v1')  

    episode_durations_list = []
    reinforce_loss_list = []

    for i in range(n_runs):
        start_time = time()

        policy_model = PolicyNetwork(input_dim=4, hidden_dim=hidden_dim, output_dim=2)  # change input_ and output_dim for gridworld env
        seed = 40 + i
        set_seeds(env, seed)

        episode_durations, reinforce_loss = run_episodes_with_SC_baseline( 
            policy_model,
            env,
            n_episodes,
            discount_factor,
            learn_rate,
            init_temp,
            stochasticity
        )

        episode_durations_list.append(episode_durations)
        reinforce_loss_list.append(reinforce_loss)
                        
        del policy_model

        end_time = time()
        h, m, s = get_running_time(end_time - start_time)

        print(f'Done with run {i+1}/{n_runs} in {f"{h} hours, " if h else ""}{f"{m} minutes and " if m else ""}{s} seconds')

    env.close()

    et = time()
    h, m, s = get_running_time(et - st)

    evals = {}
    evals['episode_durations'] = episode_durations_list
    evals['reinforce_loss'] = reinforce_loss_list

    pkl.dump(evals, open(eval_file, 'wb'))

    print(f'Done with runs in {f"{h} hours, " if h else ""}{f"{m} minutes and " if m else ""}{s} seconds')


def run_no_baseline(stochasticity, n_runs, n_episodes):
    # no baseline
    dir_path = os.path.dirname(os.path.realpath(__file__))
    best_settings_file = dir_path+f'/cart_pole_parameter_search/s{stochasticity}_no_baseline_best_settings.pkl'
    eval_file = f'cart_evals/s{stochasticity}_no_baseline_.pkl'

    with open(best_settings_file, 'rb') as pickle_file:
        best_settings = pkl.load(pickle_file)
    discount_factor = best_settings['discount_factor'] 
    learn_rate = best_settings['learn_rate']
    hidden_dim = best_settings['hidden_dim']
    init_temp = best_settings['init_temp']

    st = time()

    # change this for learned baseline
    print(f'Run settings: baseline=run_episodes_no_baseline, discount_factor={discount_factor}, learn_rate={learn_rate}, hidden_dim={hidden_dim}, init_temp={init_temp}')

    # initialize the environment
    env = gym.make('CartPole-v1')   # <---------- change this!

    episode_durations_list = []
    reinforce_loss_list = []

    for i in range(n_runs):
        start_time = time()

        policy_model = PolicyNetwork(input_dim=4, hidden_dim=hidden_dim, output_dim=2)  # change input_ and output_dim for gridworld env
        seed = 40 + i
        set_seeds(env, seed)

        episode_durations, reinforce_loss = run_episodes_no_baseline( 
            policy_model,
            env,
            n_episodes,
            discount_factor,
            learn_rate,
            init_temp,
            stochasticity
        )

        episode_durations_list.append(episode_durations)
        reinforce_loss_list.append(reinforce_loss)

        del policy_model

        end_time = time()
        h, m, s = get_running_time(end_time - start_time)

        print(f'Done with run {i+1}/{n_runs} in {f"{h} hours, " if h else ""}{f"{m} minutes and " if m else ""}{s} seconds')

    env.close()

    et = time()
    h, m, s = get_running_time(et - st)

    evals = {}
    evals['episode_durations'] = episode_durations_list
    evals['reinforce_loss'] = reinforce_loss_list

    pkl.dump(evals, open(eval_file, 'wb'))

    print(f'Done with runs in {f"{h} hours, " if h else ""}{f"{m} minutes and " if m else ""}{s} seconds')

# Choose what you want to run by uncommenting
#run_no_baseline(stochasticity, n_runs, n_episodes)
#run_learned_baseline(stochasticity, n_runs, n_episodes)
#run_selfcritic_baseline(stochasticity, n_runs, n_episodes)
