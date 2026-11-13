import math
from typing import Any, Dict, List, Optional, Type, Union

import gym
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


__all__ = ["BatchRenorm", "BatchRenorm1d", "NoisyLinear", "CustomCNN", "create_mlp"]


class BatchRenorm(torch.nn.Module):
    """
    BatchRenorm Module (https://arxiv.org/abs/1702.03275).
    Adapted to Pytorch from
    https://github.com/araffin/sbx/blob/master/sbx/common/jax_layers.py

    BatchRenorm is an improved version of vanilla BatchNorm. Contrary to BatchNorm,
    BatchRenorm uses the running statistics for normalizing the batches after a warmup phase.
    This makes it less prone to suffer from "outlier" batches that can happen
    during very long training runs and, therefore, is more robust during long training runs.

    During the warmup phase, it behaves exactly like a BatchNorm layer. After the warmup phase,
    the running statistics are used for normalization. The running statistics are updated during
    training mode. During evaluation mode, the running statistics are used for normalization but
    not updated.

    :param num_features: Number of features in the input tensor.
    :param eps: A value added to the variance for numerical stability.
    :param momentum: The value used for the ra_mean and ra_var (running average) computation.
        It controls the rate of convergence for the batch renormalization statistics.
    :param affine: A boolean value that when set to True, this module has learnable
            affine parameters. Default: True
    :param warmup_steps: Number of warum steps that are performed before the running statistics
            are used for normalization. During the warump phase, the batch statistics are used.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 0.001,
        momentum: float = 0.01,
        affine: bool = True,
        warmup_steps: int = 100_000,
    ):
        super().__init__()
        # Running average mean and variance
        self.register_buffer("ra_mean", torch.zeros(num_features, dtype=torch.float))
        self.register_buffer("ra_var", torch.ones(num_features, dtype=torch.float))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))
        self.scale = torch.nn.Parameter(torch.ones(num_features, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.zeros(num_features, dtype=torch.float))

        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        # Clip scale and bias of the affine transform
        self.rmax = 3.0
        self.dmax = 5.0
        self.warmup_steps = warmup_steps

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor.

        :param x: Input tensor
        :return: Normalized tensor.
        """

        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            batch_std = (batch_var + self.eps).sqrt()

            # Use batch statistics during initial warm up phase.
            # Note: in the original paper, after some warmup phase (batch norm phase of 5k steps)
            # the constraints are linearly relaxed to r_max/d_max over 40k steps
            # Here we only have a warmup phase
            if self.steps > self.warmup_steps:  # type: ignore[operator]

                running_std = (self.ra_var + self.eps).sqrt()  # type: ignore[operator, union-attr]
                # scale
                r = (batch_std / running_std).detach()
                r = r.clamp(1 / self.rmax, self.rmax)
                # bias
                d = ((batch_mean - self.ra_mean) / running_std).detach()  # type: ignore[operator]
                d = d.clamp(-self.dmax, self.dmax)

                # BatchNorm normalization, using minibatch stats and running average stats
                custom_mean = batch_mean - d * batch_var.sqrt() / r
                custom_var = batch_var / (r**2)

            else:
                custom_mean, custom_var = batch_mean, batch_var

            # Update Running Statistics
            self.ra_mean += self.momentum * (batch_mean.detach() - self.ra_mean)  # type: ignore[operator]
            self.ra_var += self.momentum * (batch_var.detach() - self.ra_var)  # type: ignore[operator]
            self.steps += 1  # type: ignore[operator, assignment]

        else:
            # Use running statistics during evaluation mode
            custom_mean, custom_var = self.ra_mean, self.ra_var  # type: ignore[assignment]

        # Normalize
        x = (x - custom_mean[None]) / (custom_var[None] + self.eps).sqrt()

        if self.affine:
            x = self.scale * x + self.bias

        return x

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, momentum={self.momentum}, "
            f"warmup_steps={self.warmup_steps}, affine={self.affine}"
        )


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() == 1:
            raise ValueError(f"Expected 2D or 3D input (got {x.dim()}D input)")


class NoisyLinear(nn.Module):
    """
    Noisy Linear

    :param input_dim: Dimension of the input vector
    :param output_dim:
    # https://github.com/Kaixhin/Rainbow/blob/master/model.py
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        std_init: float = 0.5,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:

        self.atom_size = 0
        if "atom_size" in kwargs:
            self.atom_size = kwargs["atom_size"]
        if self.atom_size > 0:
            out_features = out_features * self.atom_size

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            th.empty((out_features, in_features), **factory_kwargs)
        )
        self.weight_sigma = nn.Parameter(
            th.empty((out_features, in_features), **factory_kwargs)
        )
        self.register_buffer(
            "weight_epsilon", th.Tensor(th.empty((out_features, in_features)))
        )

        self.bias_mu = nn.Parameter(th.empty((out_features), **factory_kwargs))
        self.bias_sigma = nn.Parameter(
            th.empty((out_features), **factory_kwargs)
        )
        self.register_buffer("bias_epsilon", th.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> th.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = th.randn(size)

        return x.sign().mul(x.abs().sqrt())


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 64,
        cnn=None,
    ):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert cnn is not None
        self.cnn = cnn

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    layer_mod: Type[nn.Module] = nn.Linear,
    layer_kwargs: Optional[Dict[str, Any]] = None,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [
            layer_mod(input_dim, net_arch[0], **layer_kwargs),
            activation_fn(),
        ]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(
            layer_mod(net_arch[idx], net_arch[idx + 1], **layer_kwargs)
        )
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(layer_mod(last_layer_dim, output_dim, **layer_kwargs))
    if squash_output:
        modules.append(nn.Tanh())
    return modules
