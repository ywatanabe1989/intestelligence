#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-10-16 10:28:24 (ywatanabe)"

import numpy as np
import torch
import torch.nn as nn


class PerceptronOrINN(nn.Module):
    """
    Three-layer neural network with customizable activation functions.

    Parameters:
    - config (dict): Configuration for the model.

    Switch between Perceptron ('sigmoid') and an intestinal data-based neural network (INN; 'intestine_simulated') by setting 'act_str' in config:
    - For Perceptron: config = {"act_str": "sigmoid", ...}
    - For INN: config = {"act_str": "intestine_simulated", ...}
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # the 1st layer
        self.fc1 = nn.Linear(config["n_fc_in"], config["n_fc_1"])
        self.act_layer_1 = Activation(
            (config["n_fc_1"],),
            config["do_resample_act_funcs"],
            self.config[f"{self.config['act_str']}_beta_0_mean"],
            self.config[f"{self.config['act_str']}_beta_0_var"],
            self.config[f"{self.config['act_str']}_beta_1_mean"],
            self.config[f"{self.config['act_str']}_beta_1_var"],
        )
        self.dropout_layer_1 = nn.Dropout(config["d_ratio_1"])

        # the 2nd layer        
        self.fc2 = nn.Linear(config["n_fc_1"], config["n_fc_2"])
        self.act_layer_2 = Activation(
            (config["n_fc_2"],),
            config["do_resample_act_funcs"],
            self.config[f"{self.config['act_str']}_beta_0_mean"],
            self.config[f"{self.config['act_str']}_beta_0_var"],
            self.config[f"{self.config['act_str']}_beta_1_mean"],
            self.config[f"{self.config['act_str']}_beta_1_var"],
        )
        
        # the last layer
        self.fc3 = nn.Linear(config["n_fc_2"], len(config["LABELS"]))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer_1(x)
        x = self.dropout_layer_1(x)
        x = self.fc2(x)
        x = self.act_layer_2(x)
        x = self.fc3(x)
        return x


class Activation(nn.Module):
    def __init__(
        self,
        shape,
        do_resample_act_funcs,
        beta_0_mean,
        beta_0_var,
        beta_1_mean,
        beta_1_var,
    ):
        super().__init__()
        self.shape = shape
        self.do_resample_act_funcs = do_resample_act_funcs
        self.beta_0_mean = beta_0_mean
        self.beta_0_var = beta_0_var
        self.beta_0_std = float(np.sqrt(beta_0_var))
        self.beta_1_mean = beta_1_mean
        self.beta_1_var = beta_1_var
        self.beta_1_std = float(np.sqrt(beta_1_var))
        self.resample()

    def resample(self):
        # Sample from a bimodal Gaussian distribution
        mix = (
            torch.rand(self.shape) < 0.5
        )  # Mixing coefficient to select which Gaussian to sample from
        self.beta_0 = torch.where(
            mix,
            torch.normal(
                mean=self.beta_0_mean,
                std=self.beta_0_std,
                size=self.shape,
            ),
            torch.normal(mean=-self.beta_0_mean, std=self.beta_0_std, size=self.shape),
        )
        self.beta_1 = torch.where(
            mix,
            torch.normal(mean=self.beta_1_mean, std=self.beta_1_std, size=self.shape),
            torch.normal(mean=-self.beta_1_mean, std=self.beta_1_std, size=self.shape),
        )

    def forward(self, x):
        if self.do_resample_act_funcs:
            self.resample()
        return 1 / (1 + torch.exp(-self.beta_0.type_as(x) * x + self.beta_1.type_as(x)))


if __name__ == "__main__":
    batch_size, n_in = 16, 28 * 28
    Xb = torch.rand(batch_size, n_in).cuda()

    model_config = {
        "act_str": "intestine_simulated",  # "sigmoid"
        "do_resample_act_funcs": False,
        "bs": batch_size,
        "n_fc_in": n_in,
        "n_fc_1": 1000,
        "n_fc_2": 1000,
        "d_ratio_1": 0.5,
        "sigmoid_beta_0_mean": 1,  # For an ordinary three-layer perceptron
        "sigmoid_beta_0_var": 0,
        "sigmoid_beta_1_mean": 0,
        "sigmoid_beta_1_var": 0,
        "intestine_simulated_beta_0_mean": 3.06,  # For INN
        "intestine_simulated_beta_0_var": 1.38,
        "intestine_simulated_beta_1_mean": 0,
        "intestine_simulated_beta_1_var": 3.23,
        "LABELS": [0, 1],  # Replace this with your actual labels
    }

    model = PerceptronOrINN(model_config).cuda()
    y = model(Xb)
    y.sum().backward()
