import numpy as np
import torch
import matplotlib.pyplot as plt

from collections import deque
from typing import Sequence, Union


def batchify_obs(obs, env, device):
    """Converts PZ style observations to batch of torch arrays."""

    # convert to torch
    obs_mob = {agent: torch.tensor(obs[agent]['obs_mob']).float().to(device)
               for agent in env.possible_agents}
    obs_fog = {agent: torch.tensor(obs[agent]['obs_fog']).float().to(device)
               for agent in env.possible_agents}
    action_masks = {agent: torch.tensor(obs[agent]['action_mask']).float().to(device)
                    for agent in env.possible_agents}

    return obs_mob, obs_fog, action_masks


def batchify(x, env, device):
    """Converts PZ style returns to batch of torch arrays."""

    # convert to torch
    x = {agent: torch.tensor(x[agent]).to(device) for agent in env.possible_agents}

    return x


def unbatchify(x, env):
    """Converts torch tensor to PZ style arguments."""

    x = {agent: x[agent].cpu().numpy() for agent in env.possible_agents}

    return x


def evaluate(env, rl_agents, device, max_recurrent_steps=10, path=None,
             num_episodes: int = 1) -> Sequence[Union[int, float]]:
    actions = {}
    actions_dict = {agent: list() for agent in env.possible_agents}
    # cumu_rewards = 0

    # Evaluate the policy for num_episode episodes
    for e in range(num_episodes):
        rewards_list = list()
        dropped_list = list()
        delay_list = list()

        # For updating rewards
        reward_indicator = np.zeros([env.n_time, env.n_iot])

        obs_fog_buff = {
            agent: deque(maxlen=max_recurrent_steps) for agent in env.possible_agents}

        obs, _ = env.reset()

        terms = [False]
        truncs = [False]

        # The state -> action -> (reward, next-state) loop
        while not any(terms) and not any(truncs):
            obs_mob, obs_fog, action_mask = batchify_obs(obs, env, device)
            for agent in env.possible_agents:
                # Save FOG_observations to buffer
                obs_fog_buff[agent].append(obs_fog[agent])

                actions[agent] = rl_agents[agent].get_action(
                    torch.unsqueeze(torch.vstack(
                                    tuple(obs_fog_buff[agent])).float().to(device), 0),
                    obs_mob[agent], action_mask=action_mask[agent], inference=True)

                actions_dict[agent].append(actions[agent].item())

            obs, rewards, terms, truncs, infos = \
                env.step(unbatchify(actions, env))
            terms = [terms[a] for a in terms]
            truncs = [truncs[a] for a in truncs]

            process_delay = env.process_delay
            unfinish_inds = env.process_delay_unfinish_ind

            for iot_index, a in enumerate(env.possible_agents):
                update_index = np.where((1 - reward_indicator[:, iot_index]) *
                                        process_delay[:, iot_index] > 0)[0]

                if len(update_index) != 0:
                    for ind in range(len(update_index)):
                        time_index = update_index[ind]

                        reward = env.get_reward(
                            process_delay[time_index, iot_index],
                            unfinish_inds[time_index, iot_index])

                        dropped_list.append(unfinish_inds[time_index, iot_index])
                        if not unfinish_inds[time_index, iot_index]:
                            delay_list.append(process_delay[time_index, iot_index])

                        reward_indicator[time_index, iot_index] = 1

                        rewards_list.append(-reward)
                        # rewards_list.append(reward)

    for a in env.possible_agents:
        count = {action: actions_dict[a].count(action) for action in range(env.n_actions)}
        print(f"{a}: {count}")

    return np.mean(rewards_list)/env.n_iot


def plot_all_marl(
   axs, train_episode_t: Sequence[int], train_rewards: Sequence[float],
   eval_episode_t: Sequence[int], eval_rewards: Sequence[float], entrophy=None,
   text=None, show=False, path=None, save=False, close=False) -> None:

    # Episode Rewards
    axs.clear()
    axs.plot(train_episode_t, train_rewards, color='red', label='train_reward')
    axs.plot(eval_episode_t, eval_rewards, color='blue', label='eval_reward')
    axs.set(title='Episode Rewards.')
    axs.set(ylabel='Avg. Cummu. Reward')
    axs.set(xlabel='Episodes')
    axs.legend(loc='lower right')

    if text is not None:
        x_min = axs.get_xlim()[0]
        y_max = axs.get_ylim()[1]
        axs.text(x_min * 1.0, y_max * 1.01, text, fontsize=14, color='Black')

    if save:
        plt.savefig(path + "learning_curves.png")

    if show:
        plt.show(block=False)
        plt.pause(0.01)

    if close:
        plt.close()


def plot_graphs(axs, train_cost, train_dropped, train_delay, show=False, save=False,
                path=None):
    x = np.arange(len(train_cost)).tolist()
    axs[0].clear()
    axs[0].plot(x, train_cost, color='red', label='Training')
    axs[0].set(title='Avg. Cost')
    axs[0].set(ylabel='Avg. Cost')
    axs[0].set(xlabel='Episode')
    axs[0].legend(loc='upper right')

    axs[1].clear()
    axs[1].plot(x, train_dropped, color='blue', label='Training')
    axs[1].set(title='Ratio of Dropped Tasks')
    axs[1].set(ylabel='Dropped Ratio')
    axs[1].set(xlabel='Episode')
    axs[1].legend(loc='upper right')

    axs[2].clear()
    axs[2].plot(x, train_delay, color='green', label='Training')
    axs[2].set(title='Avg. Task Delay')
    axs[2].set(ylabel='Avg. Delay (Sec)')
    axs[2].set(xlabel='Episode')
    axs[2].legend(loc='upper right')

    if save:
        plt.savefig(path + "plots/learning_curves.png")

        with open(path + 'plots/avg_cost.npy', 'wb') as f:
            np.save(f, np.array(train_cost))

        with open(path + 'plots/dropped_ratio.npy', 'wb') as f:
            np.save(f, np.array(train_dropped))

        with open(path + 'plots/avg_delay.npy', 'wb') as f:
            np.save(f, np.array(train_delay))

    if show:
        plt.show(block=False)
        plt.pause(0.01)
