from typing import Any, Dict, List, Optional, Type

import gym
import torch as th

from stable_baselines3.common.policies import BasePolicy
from torch import nn
from torch.nn.functional import softmax

from sb3_contrib.common.torch_layers import NoisyLinear, create_mlp



class QNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        layer_mod: Optional[nn.Module] = nn.Linear,
        layer_kwargs: Optional[Dict[str, Any]] = {},
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.normalize_images = normalize_images
        self.layer_mod = layer_mod
        self.layer_kwargs = layer_kwargs
        action_dim = self.action_space.n  # number of actions
        q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn, layer_mod=layer_mod, layer_kwargs=layer_kwargs)
        self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        return self.q_net(self.extract_features(obs))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                layer_mod=self.layer_mod,
                layer_kwargs=self.layer_kwargs
            )
        )
        return data

class DuelingQNetwork(BasePolicy):
    """
    Dueling Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        support: Optional[th.Tensor] = None,
        layer_mod: Optional[nn.Module] = nn.Linear,
        layer_kwargs: Optional[Dict[str, Any]] = {},
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            # default value taken from https://doi.org/10.48550/arXiv.1511.06581
            net_arch = [512]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.normalize_images = normalize_images
        self.action_dim = self.action_space.n  # number of actions
        self.support = support
        self.layer_mod = layer_mod
        self.layer_kwargs = layer_kwargs
            
        value_net = create_mlp(
            self.features_dim,
            1,
            self.net_arch,
            self.activation_fn,
            layer_mod=layer_mod,
            layer_kwargs=layer_kwargs
        )
        advantage_net = create_mlp(
            self.features_dim,
            self.action_dim,
            self.net_arch,
            self.activation_fn,
            layer_mod=layer_mod,
            layer_kwargs=layer_kwargs
        )

        self.value_stream = nn.Sequential(*value_net)
        self.advantage_stream = nn.Sequential(*advantage_net)

    def forward(
        self, obs: th.Tensor
    ) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """

        if self.atom_size > 1:
            dist = self.dist(obs)
            q_values = th.sum(dist * self.support, dim=2)
        else:
            features = self.extract_features(obs)
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            q_values = (
                value + advantages - advantages.mean(dim=-1, keepdim=True)
            )

        return q_values

    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = True,
    ) -> th.Tensor:
        q_values = self(observation)

        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                support=self.support,
                layer_mod=self.layer_mod,
                layer_kwargs=self.layer_kwargs
            )
        )
        return data

    def dist(self, obs: th.Tensor) -> th.Tensor:
        """Get distribution for atoms."""
        features = self.extract_features(obs)
        advantage = self.advantage_stream(features).view(
            -1, self.action_dim, self.atom_size
        )
        value = self.value_stream(features).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        for layer in self.value_stream.modules():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_stream.modules():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
