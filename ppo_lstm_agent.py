import numpy as np

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class NNAgent(nn.Module):
    def __init__(self, recurrent_inp_size: int, fc_inp_size: int, num_actions: int,
                 recurrent_hidden_size: int = 20, hl_1_size: int = 20, hl_2_size: int = 20):
        super().__init__()

        self.recurrent_layer = nn.LSTM(recurrent_inp_size, recurrent_hidden_size,
                                       batch_first=True)

        self.network = nn.Sequential(
            self._layer_init(nn.Linear(fc_inp_size + recurrent_hidden_size, hl_1_size)),
            nn.ReLU(),
            self._layer_init(nn.Linear(hl_1_size, hl_2_size)),
            nn.ReLU(),
        )

        self.actor = self._layer_init(nn.Linear(hl_2_size, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(hl_2_size, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, recurrent_inp, fc_inp, fc_out=None):
        if fc_out is None:
            recurrent_out = torch.squeeze(
                self.recurrent_layer(recurrent_inp)[0][:, -1, :], 0)
            # fc_out = self.network(torch.hstack((fc_inp, recurrent_out)))
            fc_out = self.network(
                torch.hstack((fc_inp, torch.squeeze(recurrent_inp[:, -1, :], 0),
                              recurrent_out)))

        return self.critic(fc_out)

    def get_action(self, recurrent_inp, fc_inp, fc_out=None, action_mask=None,
                   inference=False):
        if fc_out is None:
            recurrent_out = torch.squeeze(
                self.recurrent_layer(recurrent_inp)[0][:, -1, :], 0)
            # fc_out = self.network(torch.hstack((fc_inp, recurrent_out)))
            fc_out = self.network(
                torch.hstack((fc_inp, torch.squeeze(recurrent_inp[:, -1, :], 0),
                              recurrent_out)))

        logits = self.actor(fc_out)

        if action_mask is not None:
            logits += torch.log(action_mask)

        probs = Categorical(logits=logits)

        if not inference:
            action = probs.sample()
        else:
            action = torch.argmax(probs.probs, dim=-1)

        return action

    def get_action_and_value(self, recurrent_inp, fc_inp, action=None, action_mask=None,
                             inference=False):
        recurrent_out = torch.squeeze(self.recurrent_layer(recurrent_inp)[0][:, -1, :], 0)
        # print(f"recurrent_inp: {recurrent_inp.shape}")
        # print(f"fc_inp: {fc_inp.shape}")
        # print(f"recurrent_out: {recurrent_out.shape}")
        # print(f"recurrent_inp[:, -1, :].shape: {recurrent_inp[:, -1, :].shape}\n")

        # fc_out = self.network(torch.hstack((fc_inp, recurrent_out)))
        fc_out = self.network(
            torch.hstack((fc_inp, torch.squeeze(recurrent_inp[:, -1, :], 0),
                          recurrent_out)))

        logits = self.actor(fc_out)

        if action_mask is not None:
            # print(f"BEFORE logits:{logits.shape}")
            # print(f"action_mask:{action_mask}")
            logits += torch.log(action_mask)
            # print(f"AFTER  logits:{logits.shape}\n")

        probs = Categorical(logits=logits)

        if action is None:
            if not inference:
                action = probs.sample()
            else:
                action = torch.argmax(probs.probs, dim=-1)
            # print(f"action.shape: {type(action)}\n")
        # else:
        #     print(f"recurrent_inp: {recurrent_inp.shape}")
        #     print(f"fc_inp: {fc_inp.shape}")
        #     print(f"recurrent_out: {recurrent_out.shape}\n")

        return action, probs.log_prob(action), probs.entropy(), self.critic(fc_out)
