import numpy as np
import random
# import time
import argparse
import matplotlib.pyplot as plt

from fog_env import FOGEnv
from utils import plot_graphs


def random_policy(env, num_episodes, show=False):
    episode_rewards = list()
    episode_dropped = list()
    episode_delay = list()

    fig, axs = plt.subplots(3, figsize=(10, 12), sharex=True)

    for episode in range(num_episodes):
        rewards_list = list()
        dropped_list = list()
        delay_list = list()

        reward_indicator = np.zeros([env.n_time, env.n_iot])
        done = False

        # INITIALIZE OBSERVATION
        observation, infos = env.reset()

        # TRAIN DRL
        while True:

            # PERFORM ACTION
            actions = np.zeros([env.n_iot])
            for iot_index, agent in enumerate(env.possible_agents):
                # print(f"observation[agent]:\n{observation[agent]}")
                obs = observation[agent]['observation'][:-env.n_fog]
                if np.sum(obs) == 0:
                    # if there is no task, action = 0 (also need to be stored)
                    actions[iot_index] = 0
                    # print("No Task")
                else:  # Follow a random action
                    actions[iot_index] = np.random.randint(env.n_actions)
                    # print(f"Random Action: {actions[iot_index]}")

            # OBSERVE THE NEXT STATE AND PROCESS DELAY (REWARD)
            observations_next, rewards, terminations, truncations, infos = \
                env.step(actions)

            process_delay = env.process_delay
            unfinish_indi = env.process_delay_unfinish_ind

            # STORE MEMORY; STORE TRANSITION IF THE TASK PROCESS DELAY IS JUST UPDATED
            for iot_index, agent in enumerate(env.possible_agents):
                update_index = np.where((1 - reward_indicator[:, iot_index]) *
                                        process_delay[:, iot_index] > 0)[0]

                if len(update_index) != 0:
                    for update_ii in range(len(update_index)):
                        time_index = update_index[update_ii]

                        reward = env.get_reward(process_delay[time_index, iot_index],
                                                unfinish_indi[time_index, iot_index])

                        dropped_list.append(unfinish_indi[time_index, iot_index])
                        if not unfinish_indi[time_index, iot_index]:
                            delay_list.append(process_delay[time_index, iot_index])

                        reward_indicator[time_index, iot_index] = 1

                        # rewards_dict[iot_index].append(-reward)
                        rewards_list.append(-reward)

                done = done or terminations[agent] or truncations[agent]

            # UPDATE OBSERVATION
            observation = observations_next

            # GAME ENDS
            if done:
                break

        avg_reward = np.mean(rewards_list)/env.n_iot
        episode_rewards.append(avg_reward)

        dropped_ratio = np.mean(dropped_list)/env.n_iot
        episode_dropped.append(dropped_ratio)

        avg_delay = np.mean(delay_list)/env.n_iot
        episode_delay.append(avg_delay)

        print(f"Episode: {episode} - Reward: {avg_reward} - Dropped: {dropped_ratio} - "
              + f"Delay: {avg_delay}")

        if episode % 10 == 0:
            plot_graphs(axs, episode_rewards, episode_dropped, episode_delay, show=show,
                        save=False)

    plot_graphs(axs, episode_rewards, episode_dropped, episode_delay, show=show,
                save=False)

    input("Completed.\nPress Enter to Finish")


def main(args):
    # Set random generator seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # GENERATE ENVIRONMENT
    env = FOGEnv(args.num_iot, args.num_fog, NUM_TIME, MAX_DELAY, args.task_arrival_prob)

    # TRAIN THE SYSTEM
    random_policy(env, args.num_episodes, args.plot)


if __name__ == "__main__":

    NUM_TIME_BASE = 100
    MAX_DELAY = 10
    NUM_TIME = NUM_TIME_BASE + MAX_DELAY

    parser = argparse.ArgumentParser(description='Random Policy for Mobile Edge Comp.')
    parser.add_argument('--num_iot', type=int, default=50,
                        help='number of IOT devices (default: 50)')
    parser.add_argument('--num_fog', type=int, default=5,
                        help='number of FOG stations (default: 5)')
    parser.add_argument('--task_arrival_prob', type=float, default=0.3,
                        help='Task Arrival Probability (default: 0.3)')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='number of training episodes (default: 1000)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--plot',  default=False, action='store_true',
                        help='plot learning curve (default: False)')
    args = parser.parse_args()

    main(args)
