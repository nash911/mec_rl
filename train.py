import numpy as np
import random
import os
import argparse
import json

from datetime import datetime
from shutil import rmtree
from typing import Mapping

import torch
import torch.optim as optim

from fog_env import FOGEnv
from ppo_lstm_agent import NNAgent
from clean_ppo import CleanPPO


def save_learning_params() -> Mapping[str, bool | int | float]:
    learning_params = dict()

    # Training
    learning_params['update_epochs'] = UPDATE_EPOCHS
    learning_params['eval_freq'] = EVAL_FREQ

    # AGENT
    learning_params['recurrent_hidden_size'] = RECURRENT_HIDDEN_SIZE
    learning_params['hl_1_size'] = HL1_SIZE
    learning_params['hl_2_size'] = HL2_SIZE
    learning_params['max_seq_len'] = MAX_SEQ_LEN

    # PPO
    learning_params['entrophy_coefficient'] = ENT_COEF
    learning_params['entrophy_decay'] = ENT_DECAY
    learning_params['value_function_coefficient'] = VF_COEF
    learning_params['clip_coefficient'] = CLIP_COEF
    learning_params['gamma'] = GAMMA
    learning_params['gae_lambda'] = GAE_LAMBDA
    learning_params['batch_size'] = BATCH_SIZE
    learning_params['normalize_advantage'] = NORM_ADV
    learning_params['clip_value_loss'] = CLIP_VLOSS
    learning_params['max_grad_norm'] = MAX_GRAD_NORM
    learning_params['target_kl'] = TARGET_KL

    # ADAM
    learning_params['learning_rate'] = LR
    learning_params['anneal_learning_rage'] = ANNEAL_LR

    return learning_params


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random generator seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create a timestamp directory to save model, parameter and log files
    training_dir = \
        ('training/' + ('' if args.path is None else args.path + '/') +
         str(datetime.now().date()) + '_' + str(datetime.now().hour).zfill(2) + '-' +
         str(datetime.now().minute).zfill(2) + '/')

    # Delete if a directory with the same name already exists
    if os.path.exists(training_dir):
        rmtree(training_dir)

    # Create empty directories for saving model, parameter and log files
    os.makedirs(training_dir)
    os.makedirs(training_dir + 'plots')
    os.makedirs(training_dir + 'results')
    os.makedirs(training_dir + 'params')
    os.makedirs(training_dir + 'learning')
    os.makedirs(training_dir + 'models')

    # Dump params to file
    with open(training_dir + 'params/params.dat', 'w') as jf:
        json.dump(vars(args), jf, indent=4)

    plot_dict = {'color': args.plot_color, 'label': args.plot_label}
    with open(training_dir + 'plots/plot_props.dat', 'w') as jf:
        json.dump(plot_dict, jf, indent=4)

    """ ENV SETUP """  # Set the environments (one for training, one for evaluation)
    train_env = FOGEnv(args.num_iot, args.num_fog, NUM_TIME, MAX_DELAY,
                       args.task_arrival_prob)
     # seed=SEED, path=train_path, save_results=True)

    num_actions = train_env.action_space(train_env.possible_agents[0]).n
    # obs_size = train_env.observation_space(train_env.possible_agents[0]).shape[0]
    obs_fog_size = \
        train_env.observation_space(train_env.possible_agents[0])['obs_fog'].shape[0]
    obs_mob_size = \
        train_env.observation_space(train_env.possible_agents[0])['obs_mob'].shape[0]

    print(f"obs_fog_size: {obs_fog_size}  --  obs_mob_size: {obs_mob_size}")

    eval_env = FOGEnv(args.num_iot, args.num_fog, NUM_TIME, MAX_DELAY,
                      args.task_arrival_prob)

    """ LEARNER SETUP FOR RL VS RL"""
    # RL Agents - Policy and Value Networks for the RL agent
    rl_agents = {agent: NNAgent(
        recurrent_inp_size=obs_fog_size, fc_inp_size=obs_mob_size+obs_fog_size,
        num_actions=num_actions, recurrent_hidden_size=RECURRENT_HIDDEN_SIZE,
        hl_1_size=HL1_SIZE, hl_2_size=HL2_SIZE).to(device)
        for agent in train_env.possible_agents}

    # Optimizer - Adam
    optimizers = {agent: optim.Adam(rl_agents[agent].parameters(), lr=LR, eps=1e-5)
                  for agent in train_env.possible_agents}

    # PPO algorithm with SelfPlay MARL
    ppo = CleanPPO(train_env=train_env, eval_env=eval_env, agents=rl_agents,
                   optimizers=optimizers, max_seq_len=MAX_SEQ_LEN,
                   episode_length=NUM_TIME, device=device)

    # Save the learning parameters for reference
    learning_params = save_learning_params()

    # Dump learning params to file
    with open(training_dir + 'learning/learning_params.dat', 'w') as jf:
        json.dump(learning_params, jf, indent=4)

    # Train the agents using SelfPlay
    ppo.train(ent_coef=ENT_COEF, ent_decay=ENT_DECAY, vf_coef=VF_COEF, gamma=GAMMA,
              clip_coef=CLIP_COEF, batch_size=BATCH_SIZE, gae_lambda=GAE_LAMBDA,
              eval_freq=EVAL_FREQ, num_episodes=args.num_episodes, anneal_lr=ANNEAL_LR,
              learning_rate=LR, update_epochs=UPDATE_EPOCHS, norm_adv=NORM_ADV,
              clip_vloss=CLIP_VLOSS, max_grad_norm=MAX_GRAD_NORM, target_kl=TARGET_KL,
              verbose=args.verbose, path=training_dir, plot=args.plot)

    # if args.training_var is not None:
    #     if args.training_var == 'lr':
    #         plot_x = args.lr
    #     elif args.training_var == 'batch_size':
    #         plot_x = args.batch_size
    #     elif args.training_var == 'optimizer':
    #         plot_x = args.optimizer
    #     elif args.training_var == 'learning_freq':
    #         plot_x = args.learning_freq
    #     elif args.training_var == 'task_arrival_prob':
    #         plot_x = args.task_arrival_prob
    #     elif args.training_var == 'num_iot':
    #         plot_x = args.num_iot
    # else:
    #     plot_x = None
    #
    # evaluate(env, iot_RL_list, 20, args.random, training_dir, plot_x)


if __name__ == "__main__":

    NUM_TIME_BASE = 100
    MAX_DELAY = 10
    NUM_TIME = NUM_TIME_BASE + MAX_DELAY

    # Training
    UPDATE_EPOCHS = 4
    EVAL_FREQ = 10

    # AGENT
    RECURRENT_HIDDEN_SIZE = 20
    HL1_SIZE = 20
    HL2_SIZE = 20
    MAX_SEQ_LEN = 10

    # PPO
    ENT_COEF = 0.1        # Entropy
    ENT_DECAY = 0.998     # Entropy decay
    VF_COEF = 0.1         # Value function coefficient
    CLIP_COEF = 0.1       # Clip coefficient, makes sure that agent doesn't drift too far
    GAMMA = 0.99          # Discount factor from Belman-equation (value of future rewards)
    GAE_LAMBDA = 0.99     # Generalized Advantage Estimate
    BATCH_SIZE = 32
    NORM_ADV = True       # Normalized advantage estimate
    CLIP_VLOSS = True     # Value network, allows to keep reward in a certain range
    MAX_GRAD_NORM = None  # Same for gradient
    TARGET_KL = None      # Smoothes out the KL-Distributions, but really hard to tune

    # ADAM
    LR = 0.001           # Learning rate
    ANNEAL_LR = False     # For decaying learning rate over time

    parser = argparse.ArgumentParser(description='DQL for Mobile Edge Computing')
    parser.add_argument('--num_iot', type=int, default=50,
                        help='number of IOT devices (default: 50)')
    parser.add_argument('--num_fog', type=int, default=5,
                        help='number of FOG stations (default: 5)')
    parser.add_argument('--task_arrival_prob', type=float, default=0.3,
                        help='Task Arrival Probability (default: 0.3)')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='number of training episodes (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for optimizer (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='rms_prop',
                        help='optimizer for updating the NN (default: rms_prop)')
    parser.add_argument('--learning_freq', type=int, default=10,
                        help='frequency of updating main/eval network (default: 10)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--plot',  default=False, action='store_true',
                        help='plot learning curve (default: False)')
    parser.add_argument('--random',  default=False, action='store_true',
                        help='follow a random policy (default: False)')
    parser.add_argument('--path', type=str, default=None,
                        help='path postfix for saving training results (default: None)')
    parser.add_argument('--training_var', type=str, default=None,
                        help='training variant: {lr, task_prob, num_iot, ...}')
    parser.add_argument('--plot_color', type=str, default='red',
                        help='plot color (default: red)')
    parser.add_argument('--plot_label', type=str, default='X',
                        help='plot label (default: X)')
    parser.add_argument('--verbose',  default=False, action='store_true',
                        help='output training logs (default: False)')
    args = parser.parse_args()

    main(args)
