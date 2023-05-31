import numpy as np
import time
import matplotlib.pyplot as plt

from typing import Optional

import torch
import torch.nn as nn

from utils import batchify_obs, batchify, unbatchify, evaluate, plot_all_marl, plot_graphs


class CleanPPO():
    def __init__(self, train_env, eval_env, agents, optimizers, max_recurrent_steps,
                 episode_length, device):

        self.device = device

        """ ENV SETUP """
        self.train_env = train_env
        self.eval_env = eval_env
        self.num_agents = len(train_env.possible_agents)
        self.num_actions = train_env.action_space(train_env.possible_agents[0]).n
        self.obs_mob_size = \
            train_env.observation_space(train_env.possible_agents[0])['obs_mob'].shape
        self.obs_fog_size = \
            train_env.observation_space(train_env.possible_agents[0])['obs_fog'].shape
        self.max_recurrent_steps = max_recurrent_steps
        self.episode_len = episode_length

        self.agents = agents
        self.optimizers = optimizers

    def eval_agent(self):
        for agent in self.train_env.possible_agents:
            self.agents[agent].eval()

    def train_agent(self):
        for agent in self.train_env.possible_agents:
            self.agents[agent].train()

    def train(self, ent_coef: float = 0.1, ent_decay: float = 1.0, vf_coef: float = 0.1,
              gamma: float = 0.99, clip_coef: float = 0.2, batch_size: float = 32,
              gae_lambda: float = 0.95, eval_freq: int = 100, num_episodes: int = 1_000,
              anneal_lr: bool = True, learning_rate: float = 2.5e-4,
              update_epochs: int = 4, norm_adv: bool = True, clip_vloss: bool = True,
              max_grad_norm: float = 0.5, target_kl: float = None, verbose: bool = False,
              path: Optional[str] = None, plot: bool = False):
        start_time = time.time()

        # Create a matlibplot canvas for plotting learning curves
        # fig, axs = plt.subplots(1, figsize=(10, 6), sharey=False, sharex=True)
        fig, axs = plt.subplots(3, figsize=(10, 12), sharex=True)

        # ALGO Logic: Storage setup
        obs_mob_buff = {
            agent: torch.zeros((self.episode_len, *self.obs_mob_size)).to(self.device)
            for agent in self.train_env.possible_agents}
        obs_fog_buff = {
            agent: torch.zeros((self.episode_len, *self.obs_fog_size)).to(self.device)
            for agent in self.train_env.possible_agents}
        action_masks_buff = {
            agent: torch.zeros((self.episode_len, self.num_actions)).to(self.device)
            for agent in self.train_env.possible_agents}
        actions_buff = {agent: torch.zeros((self.episode_len)).to(self.device)
                        for agent in self.train_env.possible_agents}
        logprobs_buff = {agent: torch.zeros((self.episode_len,)).to(self.device)
                         for agent in self.train_env.possible_agents}
        rewards_buff = {agent: torch.zeros((self.episode_len,)).to(self.device)
                        for agent in self.train_env.possible_agents}
        dones_buff = {agent: torch.zeros((self.episode_len,)).to(self.device)
                      for agent in self.train_env.possible_agents}
        values_buff = {agent: torch.zeros((self.episode_len+1,)).to(self.device)
                       for agent in self.train_env.possible_agents}  # Add for T+1

        # For storing plotting data
        train_rewards = list()
        train_episode_t = list()
        train_dropped = list()
        train_delay = list()

        eval_rewards = list()
        eval_episode_t = list()
        eval_dropped_ratios = list()
        eval_delays = list()

        entrophy_list = list()

        # For learning-rate annealing
        learning_episodes = 0

        best_eval_rewards = np.inf

        # Rollout-Store-Optimize loop
        for episode in range(1, num_episodes + 1):
            rewards_list = list()
            dropped_list = list()
            delay_list = list()

            # For updating rewards
            reward_indicator = np.zeros([self.train_env.n_time, self.train_env.n_iot])

            # Reset Env at the beginning of each episode
            new_obs, _ = self.train_env.reset()

            # Extract the most recent observations and action-masks
            next_obs_mob, next_obs_fog, next_action_mask = \
                batchify_obs(new_obs, self.train_env, self.device)

            # Init variables for episode loop
            end_step = self.episode_len - 1
            # total_episodic_return = 0

            # The episode loop - Rollout and collect transitions
            for step in range(0, self.episode_len):
                actions = {}
                logprobs = {}
                values = {}

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    for agent in self.train_env.possible_agents:
                        # Save FOG_observations to rollout buffer
                        obs_fog_buff[agent][step] = next_obs_fog[agent]

                        start_idx = max(0, step - self.max_recurrent_steps)
                        end_idx = max(step, 1)
                        recurrent_inp = torch.unsqueeze(
                            obs_fog_buff[agent][start_idx:end_idx], 0)
                        # print(f"step: {step} -- start_idx: {start_idx} -- recurrent_inp.shape: {recurrent_inp.shape}")

                        # Get the action, logp, and value for each learning-agent
                        actions[agent], logprobs[agent], _, values[agent] = \
                            self.agents[agent].get_action_and_value(
                                recurrent_inp=recurrent_inp, fc_inp=next_obs_mob[agent],
                                action_mask=next_action_mask[agent])

                # Perform one-step of the simulation
                new_obs, rewards, terms, truncs, info = \
                    self.train_env.step(unbatchify(actions, self.train_env))

                # # Batchify rewards
                # rewards = batchify(rewards, self.train_env, self.device)

                # Check if the episode has terminated or truncated
                done = any([terms[a] for a in terms]) or any([truncs[a] for a in truncs])
                terms = batchify(terms, self.train_env, self.device)

                process_delay = self.train_env.process_delay
                unfinish_inds = self.train_env.process_delay_unfinish_ind

                # Store transitions in the learning-agent's buffer
                for iot_index, agent in enumerate(self.train_env.possible_agents):
                    obs_mob_buff[agent][step] = next_obs_mob[agent]
                    # obs_fog_buff[agent][step] = next_obs_fog[agent]
                    action_masks_buff[agent][step] = next_action_mask[agent]
                    actions_buff[agent][step] = actions[agent]
                    logprobs_buff[agent][step] = logprobs[agent]
                    # rewards_buff[agent][step] = rewards[agent]
                    dones_buff[agent][step] = terms[agent]
                    values_buff[agent][step] = values[agent]

                    # # Keep track of cummulative episodic reward
                    # total_episodic_return += rewards[agent]

                    update_index = np.where((1 - reward_indicator[:, iot_index]) *
                                            process_delay[:, iot_index] > 0)[0]

                    if len(update_index) != 0:
                        for ind in range(len(update_index)):
                            time_index = update_index[ind]

                            reward = self.train_env.get_reward(
                                process_delay[time_index, iot_index],
                                unfinish_inds[time_index, iot_index])

                            # TODO: Maybe revert this?
                            rewards_buff[agent][time_index] = reward
                            # rewards_buff[agent][time_index] = np.exp(reward)

                            dropped_list.append(unfinish_inds[time_index, iot_index])
                            if not unfinish_inds[time_index, iot_index]:
                                delay_list.append(process_delay[time_index, iot_index])

                            reward_indicator[time_index, iot_index] = 1

                            rewards_list.append(-reward)
                            # rewards_list.append(reward)

                # Bachify observations for the next iteration
                next_obs_mob, next_obs_fog, next_action_mask = \
                    batchify_obs(new_obs, self.train_env, self.device)

                # If end of episode (terminated or truncated)
                if done:
                    # Set end-of-episode step and exit out of the episode loop
                    end_step = step + 1
                    learning_episodes += 1

                    with torch.no_grad():
                        # Calculate and save next_values to rollout buffer
                        for agent in self.train_env.possible_agents:
                            start_idx = max(0, step - self.max_recurrent_steps)
                            recurrent_inp = torch.unsqueeze(
                                obs_fog_buff[agent][start_idx:step], 0)

                            values = self.agents[agent].get_value(recurrent_inp,
                                                                  next_obs_mob[agent])
                            values_buff[agent][step+1] = values
                    break

            train_rewards.append(np.mean(rewards_list)/self.train_env.n_iot)
            train_episode_t.append(episode)

            dropped_ratio = np.mean(dropped_list)/self.train_env.n_iot
            train_dropped.append(dropped_ratio)

            avg_delay = np.mean(delay_list)/self.train_env.n_iot
            train_delay.append(avg_delay)

            # Evaluate Policy
            if (episode - 1) % eval_freq == 0 or episode == num_episodes:
                self.eval_agent()
                eval_reward, eval_dropped, eval_delay = evaluate(
                    env=self.train_env, rl_agents=self.agents, device=self.device)
                self.train_agent()

                eval_rewards.append(eval_reward)
                eval_episode_t.append(episode)
                eval_dropped_ratios.append(eval_dropped)
                eval_delays.append(eval_delay)

                if eval_reward <= best_eval_rewards:
                    best_eval_rewards = eval_reward
                    saved_model_txt = "Best Model Saved @ Episode %d" % episode
                    for agent in self.train_env.possible_agents:
                        torch.save(self.agents[agent].state_dict(),
                                   path + 'models/' + f'best_agent_{agent}.pth')

                # Save plot data to file
                with open(path + 'plots/train_rewards.npy', 'wb') as f:
                    np.save(f, np.array(train_rewards))

                with open(path + 'plots/eval_rewards.npy', 'wb') as f:
                    np.save(f, np.array(eval_rewards))

                # Plot learning curves
                # plot_all_marl(axs, train_episode_t, train_rewards, eval_episode_t,
                #               eval_rewards, entrophy_list, text=saved_model_txt,
                #               show=plot, path=path, save=True)
                plot_graphs(axs, train_episode_t, eval_episode_t, train_rewards,
                            eval_rewards, train_dropped, eval_dropped_ratios, train_delay,
                            eval_delays, text=saved_model_txt, show=plot, path=path,
                            save=True)

            # bootstrap returns if not done
            advantages = {}
            returns = {}
            with torch.no_grad():
                for a, agent in enumerate(self.train_env.possible_agents):
                    advantages[agent] = \
                        torch.zeros_like(rewards_buff[agent]).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(end_step)):
                        delta = rewards_buff[agent][t] + \
                            (gamma * values_buff[agent][t + 1] *
                             (1 - dones_buff[agent][t])) - values_buff[agent][t]

                        advantages[agent][t] = lastgaelam = delta + \
                            (gamma * gae_lambda * (1 - dones_buff[agent][t]) * lastgaelam)

                    # Ensure to exclude T+1 value from value buffer
                    returns[agent] = advantages[agent] + values_buff[agent][:-1]

            # Annealing the rate if instructed to do so
            if anneal_lr:
                frac = 1.0 - (learning_episodes - 1.0) / num_episodes
                lrnow = frac * learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # Decay Entrophy Coefficient if instructed to do so
            if ent_decay is not None:
                ent_coef *= ent_decay
            entrophy_list.append(ent_coef)

            # Optimizing the policy and value network
            clip_fracs = []
            for agent in self.train_env.possible_agents:
                # Get the rollouts for the previous episode for the current agent
                b_obs_mob = obs_mob_buff[agent][:end_step]
                b_obs_fog = obs_fog_buff[agent][:end_step]
                b_logprobs = logprobs_buff[agent][:end_step]
                b_actions = actions_buff[agent][:end_step]
                b_action_masks = action_masks_buff[agent][:end_step]
                b_advantages = advantages[agent][:end_step]
                b_returns = returns[agent][:end_step]
                b_values = values_buff[agent][:end_step]

                # b_inds = np.arange(len(b_obs_mob))
                b_inds = np.where(torch.sum(b_obs_mob, axis=-1) != 0)[0]
                # print(f"AFTER b_inds:{b_inds}")

                b_size = (batch_size + 1 if len(b_inds) % batch_size == 1 else batch_size)

                for epoch in range(update_epochs):
                    np.random.shuffle(b_inds)
                    # for start in range(0, len(b_obs_mob), batch_size):
                    #     end = start + batch_size
                    for start in range(0, len(b_inds), b_size):
                        end = start + b_size
                        mb_inds = b_inds[start:end]

                        # print(f"epoch: {epoch} // start: {start}  --  end: {end}")

                        recurrent_inp = torch.zeros((len(mb_inds),
                                                     self.max_recurrent_steps,
                                                     self.train_env.n_lstm_state)
                                                    ).to(self.device)

                        # print(f"recurrent_inp.shape: {recurrent_inp.shape}")

                        for i, ind in enumerate(mb_inds):
                            start_idx = max(0, ind - self.max_recurrent_steps)
                            end_idx = max(1, ind)
                            #rec_inps = obs_fog_buff[agent][start_idx:end_idx]
                            rec_inps = b_obs_fog[start_idx:end_idx]
                            # print(f"rec_inps.shape: {rec_inps.shape} -- len(b_obs_mob): {len(b_obs_mob)}")
                            # print(f"ind: {ind} -- start_idx: {start_idx}")
                            recurrent_inp[i, -rec_inps.shape[0]:, :] = rec_inps

                        _, newlogprob, entropy, newvalue = \
                            self.agents[agent].get_action_and_value(
                                recurrent_inp=recurrent_inp,
                                fc_inp=b_obs_mob[mb_inds],
                                action=b_actions.long()[mb_inds],
                                action_mask=b_action_masks[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clip_fracs += [
                                ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                            ]

                        mb_advantages = b_advantages[mb_inds]

                        # Normalize advantage if instructed to do so
                        if norm_adv:
                            mb_advantages = ((mb_advantages - mb_advantages.mean()) /
                                             (mb_advantages.std() + 1e-8))

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(
                            ratio, 1 - clip_coef, 1 + clip_coef
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        if clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -clip_coef,
                                clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - (ent_coef * entropy_loss) + (v_loss * vf_coef)

                        self.optimizers[agent].zero_grad()
                        loss.backward()

                        if max_grad_norm is not None:
                            nn.utils.clip_grad_norm_(self.agents[agent].parameters(),
                                                     max_grad_norm)

                        self.optimizers[agent].step()

                    if target_kl is not None:
                        if approx_kl > target_kl:
                            break

            # Concatinate values and returns of all agents in the recent rollout
            b_values = torch.cat(
                [values_buff[agent][:end_step] for agent in
                 self.train_env.possible_agents], axis=0)
            b_returns = torch.cat(
                [returns[agent][:end_step] for agent in self.train_env.possible_agents],
                axis=0)

            # Calculate Explained Variance for logging
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if verbose:
                print(f"Training episode {episode}")
                print(f"Episodic Return: {train_rewards[-1]}")
                print(f"Episode Length: {end_step + 1}")
                print("")
                print(f"Value Loss: {v_loss.item()}")
                print(f"Policy Loss: {pg_loss.item()}")
                print(f"Old Approx KL: {old_approx_kl.item()}")
                print(f"Approx KL: {approx_kl.item()}")
                print(f"Clip Fraction: {np.mean(clip_fracs)}")
                print(f"Explained Variance: {explained_var.item()}")
                print("\n-------------------------------------------\n")

        end_time = time.time()
        print("\nTraining Time: %.2f(s)" % (end_time - start_time))
        input("Completed training.\nPress Enter to start the final evaluation")

        # Save final models (Policy and Value Networks) for evaluation
        for agent in self.train_env.possible_agents:
            torch.save(self.agents[agent].state_dict(), path + 'models/' +
                       f'final_agent_{agent}.pth')

        # # """ RENDER THE FINAL POLICY """
        # self.eval_agent()
        # with torch.no_grad():
        #     actions = {}
        #     for episode in range(1):
        #         obs, _ = self.eval_env.reset()
        #         terms = [False]
        #         truncs = [False]
        #         eval_reward = 0
        #
        #         while not any(terms) and not any(truncs):
        #             obs, action_mask = batchify_obs(obs, self.train_env, self.device)
        #
        #             # Get action for the learning-agent
        #             for agent in self.eval_env.possible_agents:
        #                 actions[agent] = self.agents[agent].get_action(
        #                     obs[agent], action_mask=action_mask[agent], inference=True)
        #
        #             obs, rewards, terms, truncs, infos = \
        #                 self.eval_env.step(unbatchify(actions, self.eval_env))
        #
        #             terms = [terms[a] for a in terms]
        #             truncs = [truncs[a] for a in truncs]
        #
        #             for a in self.eval_env.possible_agents:
        #                 eval_reward += rewards[a]
        #
        #         print("\nFinal Evaluation Avg. Reward: " +
        #               f"{eval_reward/self.episode_len/self.num_agents}")
