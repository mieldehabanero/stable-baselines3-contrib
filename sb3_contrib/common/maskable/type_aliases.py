from typing import NamedTuple

import torch as th

from stable_baselines3.common.type_aliases import ReplayBufferSamples, TensorDict

class MaskableRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor


class MaskableDictRolloutBufferSamples(MaskableRolloutBufferSamples):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor

class MaskableReplayBufferSamples(ReplayBufferSamples):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    action_masks: th.Tensor
    next_action_masks: th.Tensor


class MaskableDictReplayBufferSamples(MaskableReplayBufferSamples):
    observations: TensorDict
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    action_masks: th.Tensor
    next_action_masks: th.Tensor