from zrth import Module, Wire, DType
from .converter import convert_method
import gymnasium as gym
import torch.nn as nn


class Env(Module, gym.Env):

    def __new__(cls, *args, **kwargs):
        # TODO: trace the init for q_values and observation sizes
        q_values = [Wire(DType.TensorFloat([2])), Wire(DType.TensorFloat([2]))]
        observation = [Wire(DType.TensorFloat([1])), Wire(DType.TensorFloat([1]))]
        reward = [Wire(DType.TensorFloat([1])), Wire(DType.TensorFloat([1]))]
        terminated = [Wire(DType.TensorBool([1])), Wire(DType.TensorBool([1]))]
        truncated = [Wire(DType.TensorBool([1])), Wire(DType.TensorBool([1]))]

        # TODO: infer
        state = [Wire(DType.TensorInt([1])), Wire(DType.TensorInt([1]))]

        args = (q_values)
        result = (observation[1], reward[1], terminated[1], truncated[1])
        reset = convert_method(cls.reset, args, result)
        step = convert_method(cls.step, args, result)

        obs = [q_values, observation, reward, terminated, truncated]
        prvt = [state]
        return super().__new__(cls, init=reset, update=step, obs=obs, prvt=prvt)

class NN(Module, nn.Module):
    
    def __new__(cls, *args, **kwargs):
        # TODO: trace the init for q_values and observation sizes
        q_values = [Wire(DType.TensorFloat([1])), Wire(DType.TensorFloat([1]))]
        observation = [Wire(DType.TensorFloat([1])), Wire(DType.TensorFloat([1]))]

        args = (observation)
        result = (q_values)
        forward = convert_method(cls.forward, args, result)

        obs = [observation, q_values]
        return super().__new__(cls, assign=forward, obs=obs)

