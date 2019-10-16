from time import time
import pickle as pkl

import gym
import numpy as np

from envs.gridworld import GridworldEnv
from models import PolicyNetwork, ValueNetwork
from utils import get_running_time, set_seeds
from run_loss_functions import run_episodes_no_baseline, run_episodes_with_learned_baseline, run_episodes_with_SC_baseline

# grid search parameters
discount_factors = [0.98, 0.985, 0.99]
learn_rates = [1e-2, 1e-3, 1e-4]
hidden_dims = [64, 128, 256]
init_temps = [1.05, 1.1, 1.15]

stochasticity = 0.0  # <---------- change this!
n_runs = 5
n_episodes = 750
grid_shape = [10, 10]

def run_learned_baseline(discount_factors, learn_rates, hidden_dims, init_temps, stochasticity, n_runs, n_episodes, grid_shape):
    # learned baseline
    best_result = np.inf
    best_settings = dict()
    results_file = f'grid_results/s{stochasticity}_learned_baseline.csv'
    best_settings_file = f'grid_results/s{stochasticity}_learned_baseline_best_settings.pkl'

    with open(results_file, 'w') as f:
        f.write('discount_factor,learn_rate_policy,learn_rate_value,hidden_dim_policy,hidden_dim_value,init_temp,result' + '\n')

    for discount_factor in discount_factors:
        for learn_rate_policy in learn_rates:
            for learn_rate_value in learn_rates:
                for hidden_dim_policy in hidden_dims:
                    for hidden_dim_value in hidden_dims: 
                        for init_temp in init_temps:
                            print('#' * 30)
                            print('#' * 9 + ' NEW SEARCH ' + '#' * 9)
                            print('#' * 30)
                            print()

                            st = time()

                            # change this for learned baseline
                            print(f'Search settings: baseline=run_episodes_with_learned_baseline, discount_factor={discount_factor}, learn_rate_policy={learn_rate_policy}, learn_rate_value={learn_rate_value}, hidden_dim_policy={hidden_dim_policy}, hidden_dim_value={hidden_dim_value}, init_temp={init_temp}')

                            # initialize the environment
                            env = GridworldEnv(shape=grid_shape) 

                            result = 0

                            for i in range(n_runs):
                                start_time = time()

                                policy_model = PolicyNetwork(input_dim=2, hidden_dim=hidden_dim_policy, output_dim=4)  # change input_ and output_dim for gridworld env
                                value_model = ValueNetwork(input_dim=2, hidden_dim=hidden_dim_value)  # change input_dim for gridworld env
                                seed = 40 + i
                                set_seeds(env, seed)

                                episode_durations, _, _ = run_episodes_with_learned_baseline( 
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
                                result += np.mean(episode_durations)

                                del policy_model
                                del value_model

                                end_time = time()
                                h, m, s = get_running_time(end_time - start_time)

                                print(f'Done with run {i+1}/{n_runs} in {f"{h} hours, " if h else ""}{f"{m} minutes and " if m else ""}{s} seconds')

                            env.close()
                            result /= n_runs

                            with open(results_file, 'a') as f:
                                    f.write(f'{discount_factor},{learn_rate_policy},{learn_rate_value},{hidden_dim_policy},{hidden_dim_value},{init_temp},{result}' + '\n')

                            et = time()
                            h, m, s = get_running_time(et - st)

                            print(f'Done with search in {f"{h} hours, " if h else ""}{f"{m} minutes and " if m else ""}{s} seconds')
                            print(f'Average number of steps per episode: {result}')

                            if result < best_result:
                                best_result = result
                                best_settings['discount_factor'] = discount_factor
                                best_settings['learn_rate_policy'] = learn_rate_policy
                                best_settings['learn_rate_value'] = learn_rate_value
                                best_settings['hidden_dim_policy'] = hidden_dim_policy
                                best_settings['hidden_dim_value'] = hidden_dim_value
                                best_settings['init_temp'] = init_temp
                                best_settings['result'] = best_result
                                  
                                pkl.dump(best_settings, open(best_settings_file, 'wb'))

                                print(f'New best result!: {result}')
                                print(f'New best settings!: {best_settings}')
                            print()


    print()
    print()
    print(f'Best settings after completing grid search: {best_settings}')


def run_selfcritic_baseline(discount_factors, learn_rates, hidden_dims, init_temps, stochasticity, n_runs, n_episodes, grid_shape):         
    # self-critic baseline
    best_result = np.inf
    best_settings = dict()
    results_file = f'grid_results/s{stochasticity}_SC_baseline.csv'
    best_settings_file = f'grid_results/s{stochasticity}_SC_baseline_best_settings.pkl'

    with open(results_file, 'w') as f:
        f.write('discount_factor,learn_rate,hidden_dim,init_temp,result' + '\n')

    for discount_factor in discount_factors:
        for learn_rate in learn_rates:
            for hidden_dim in hidden_dims:
                for init_temp in init_temps:
                    print('#' * 30)
                    print('#' * 9 + ' NEW SEARCH ' + '#' * 9)
                    print('#' * 30)
                    print()

                    st = time()

                    # change this for learned baseline
                    print(f'Search settings: baseline=run_episodes_with_SC_baseline, discount_factor={discount_factor}, learn_rate={learn_rate}, hidden_dim={hidden_dim}, init_temp={init_temp}')

                    # initialize the environment
                    env = GridworldEnv(shape=grid_shape) 

                    result = 0

                    for i in range(n_runs):
                        start_time = time()

                        policy_model = PolicyNetwork(input_dim=2, hidden_dim=hidden_dim, output_dim=4)  # change input_ and output_dim for gridworld env
                        seed = 40 + i
                        set_seeds(env, seed)

                        episode_durations, _ = run_episodes_with_SC_baseline( 
                            policy_model,
                            env,
                            n_episodes,
                            discount_factor,
                            learn_rate,
                            init_temp,
                            stochasticity
                        )
                        result += np.mean(episode_durations)

                        del policy_model

                        end_time = time()
                        h, m, s = get_running_time(end_time - start_time)

                        print(f'Done with run {i+1}/{n_runs} in {f"{h} hours, " if h else ""}{f"{m} minutes and " if m else ""}{s} seconds')

                    env.close()
                    result /= n_runs

                    with open(results_file, 'a') as f:
                            f.write(f'{discount_factor},{learn_rate},{hidden_dim},{init_temp},{result}' + '\n')

                    et = time()
                    h, m, s = get_running_time(et - st)

                    print(f'Done with search in {f"{h} hours, " if h else ""}{f"{m} minutes and " if m else ""}{s} seconds')
                    print(f'Average number of steps per episode: {result}')

                    if result < best_result:
                        best_result = result
                        best_settings['discount_factor'] = discount_factor
                        best_settings['learn_rate'] = learn_rate
                        best_settings['hidden_dim'] = hidden_dim
                        best_settings['init_temp'] = init_temp
                        best_settings['result'] = best_result

                        pkl.dump(best_settings, open(best_settings_file, 'wb'))

                        print(f'New best result!: {result}')
                        print(f'New best settings!: {best_settings}')
                    print()


    print()
    print()
    print(f'Best settings after completing grid search: {best_settings}')
                      

def run_no_baseline(discount_factors, learn_rates, hidden_dims, init_temps, stochasticity, n_runs, n_episodes, grid_shape):
    # no baseline
    best_result = np.inf
    best_settings = dict()
    results_file = f'grid_results/s{stochasticity}_no_baseline.csv'
    best_settings_file = f'grid_results/s{stochasticity}_no_baseline_best_settings.pkl'

    with open(results_file, 'w') as f:
        f.write('discount_factor,learn_rate,hidden_dim,init_temp,result' + '\n')

    for discount_factor in discount_factors:
        for learn_rate in learn_rates:
            for hidden_dim in hidden_dims:
                for init_temp in init_temps:
                    print('#' * 30)
                    print('#' * 9 + ' NEW SEARCH ' + '#' * 9)
                    print('#' * 30)
                    print()

                    st = time()

                    # change this for learned baseline
                    print(f'Search settings: baseline=run_episodes_no_baseline, discount_factor={discount_factor}, learn_rate={learn_rate}, hidden_dim={hidden_dim}, init_temp={init_temp}')

                    # initialize the environment
                    env = GridworldEnv(shape=grid_shape)  # <---------- change this!

                    result = 0

                    for i in range(n_runs):
                        start_time = time()

                        policy_model = PolicyNetwork(input_dim=2, hidden_dim=hidden_dim, output_dim=4)  # change input_ and output_dim for gridworld env
                        seed = 40 + i
                        set_seeds(env, seed)

                        episode_durations, _ = run_episodes_no_baseline( 
                            policy_model,
                            env,
                            n_episodes,
                            discount_factor,
                            learn_rate,
                            init_temp,
                            stochasticity
                        )
                        result += np.mean(episode_durations)

                        del policy_model

                        end_time = time()
                        h, m, s = get_running_time(end_time - start_time)

                        print(f'Done with run {i+1}/{n_runs} in {f"{h} hours, " if h else ""}{f"{m} minutes and " if m else ""}{s} seconds')

                    env.close()
                    result /= n_runs

                    with open(results_file, 'a') as f:
                            f.write(f'{discount_factor},{learn_rate},{hidden_dim},{init_temp},{result}' + '\n')

                    et = time()
                    h, m, s = get_running_time(et - st)

                    print(f'Done with search in {f"{h} hours, " if h else ""}{f"{m} minutes and " if m else ""}{s} seconds')
                    print(f'Average number of steps per episode: {result}')

                    if result < best_result:
                        best_result = result
                        best_settings['discount_factor'] = discount_factor
                        best_settings['learn_rate'] = learn_rate
                        best_settings['hidden_dim'] = hidden_dim
                        best_settings['init_temp'] = init_temp
                        best_settings['result'] = best_result

                        pkl.dump(best_settings, open(best_settings_file, 'wb'))

                        print(f'New best result!: {result}')
                        print(f'New best settings!: {best_settings}')
                    print()


    print()
    print()
    print(f'Best settings after completing grid search: {best_settings}')

# Choose what you wann run by uncommenting
#run_no_baseline(discount_factors, learn_rates, hidden_dims, init_temps, stochasticity, n_runs, n_episodes, grid_shape)
#run_learned_baseline(discount_factors, learn_rates, hidden_dims, init_temps, stochasticity, n_runs, n_episodes, grid_shape)
#run_selfcritic_baseline(discount_factors, learn_rates, hidden_dims, init_temps, stochasticity, n_runs, n_episodes, grid_shape)