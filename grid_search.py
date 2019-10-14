from time import time

import gym
import numpy as np

from models import PolicyNetwork, ValueNetwork
from utils import get_running_time, set_seeds
from run_loss_functions import run_episodes_no_baseline, run_episodes_with_learned_baseline, run_episodes_with_SC_baseline

discount_factors = [0.98, 0.985, 0.99]
learn_rates = [1e-2, 1e-3, 1e-4]
hidden_dims = [64, 128, 256]
init_temps = [1.05, 1.1, 1.15]

stochasticity = 0.0  # change this!
n_runs = 5
n_episodes = 750
best_result = 0

for discount_factor in discount_factors:
    for learn_rate in learn_rates:  # add extra learn_rate loop for learned baseline
        for hidden_dim in hidden_dims:  # add extra hidden_dim loop for learned baseline
            for init_temp in init_temps:
                print('#' * 30)
                print('#' * 9 + ' NEW SEARCH ' + '#' * 9)
                print('#' * 30)
                print()
                # change this for learned baseline
                print(f'Search settings: discount_factor={discount_factor}, learn_rate={learn_rate}, hidden_dim={hidden_dim}, init_temp={init_temp}')

                # init the env
                env = gym.make('CartPole-v1')  # change this for gridworld env
                result = 0

                for i in range(n_runs):
                    start_time = time()

                    policy_model = PolicyNetwork(input_dim=4, hidden_dim=hidden_dim, output_dim=2)  # change input_ and output_dum for gridworld env
                    # value_model = PolicyNetwork(input_dim=4, hidden_dim=hidden_dim_value)  # uncomment this for learned baseline, change input_dim for gridworld env
                    seed = 40 + i
                    set_seeds(env, seed)

                    episode_durations, _ = run_episodes_no_baseline(  # change this for different baselines
                        policy_model,
                        # value_model,  # uncomment this for learned baseline
                        env,
                        n_episodes,
                        discount_factor,
                        learn_rate,
                        init_temp,
                        stochasticity
                    )
                    result += np.mean(episode_durations)

                    del policy_model
                    # del value_model  # uncomment this for learned baseline

                    end_time = time()
                    h, m, s = get_running_time(end_time - start_time)

                    print(f'Done with run {i+1}/{n_runs} in {f"{h} hours, " if h else ""}{f"{m} minutes and " if m else ""}{s} seconds')

                env.close()
                result /= n_runs
                print(f'Average number of episodes: {result}')

                if result > best_result:
                    print(f'New best result!: {result}')
                    best_settings = (discount_factor, learn_rate, hidden_dim, init_temp)
                    print(f'New best settings!: {best_settings}')
                print()

print()
print()
print(f'Best settings after grid search: {best_settings}')
