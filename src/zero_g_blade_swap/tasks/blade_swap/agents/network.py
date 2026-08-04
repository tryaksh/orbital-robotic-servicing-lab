"""Custom RL-Games networks used by the blade-swap teacher and vision student.

This module deliberately has no Isaac Sim imports.  It can therefore be imported by
the offline behavioural-cloning script and by lightweight unit tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

try:
    from rl_games.algos_torch import model_builder, network_builder
except ImportError:  # lets the offline actor be imported before RL-Games is installed
    model_builder = None
    network_builder = None


def _activation(name: str) -> nn.Module:
    activations = {"elu": nn.ELU, "relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation {name!r}; choose one of {sorted(activations)}") from exc


def _shape(value: Any) -> tuple[int, ...]:
    """Convert a Gym space, torch size, or sequence to a concrete shape."""
    value = getattr(value, "shape", value)
    if value is None:
        raise ValueError("RL-Games supplied an observation with no shape")
    return tuple(int(item) for item in value)


def _shape_dict(value: Any) -> dict[str, tuple[int, ...]]:
    if hasattr(value, "spaces"):
        value = value.spaces
    if not isinstance(value, Mapping):
        raise TypeError(
            "The vision policy requires dictionary observations. Set "
            "params.env.concate_obs_groups=false and expose the 'proprio' and 'rgb' groups."
        )
    return {str(key): _shape(item) for key, item in value.items()}


def _mlp(input_dim: int, units: Sequence[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for output in units:
        layers.extend((nn.Linear(current, int(output)), _activation(activation)))
        current = int(output)
    return nn.Sequential(*layers)


def _initialize(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class VisionActor(nn.Module):
    """Three-layer image encoder fused with a proprioceptive encoder."""

    def __init__(
        self,
        observation_shapes: Mapping[str, Sequence[int]],
        actions_num: int,
        rgb_key: str = "rgb",
        vector_keys: Sequence[str] = ("proprio",),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.rgb_key = rgb_key
        self.vector_keys = tuple(vector_keys)
        shapes = {key: tuple(int(item) for item in value) for key, value in observation_shapes.items()}
        missing = [key for key in (self.rgb_key, *self.vector_keys) if key not in shapes]
        if missing:
            raise KeyError(f"Vision observation is missing groups {missing}; available groups: {sorted(shapes)}")

        rgb_shape = shapes[self.rgb_key]
        if len(rgb_shape) != 3:
            raise ValueError(f"RGB group must be HWC or CHW, received shape {rgb_shape}")
        channels = rgb_shape[-1] if rgb_shape[-1] in (1, 3, 4) else rgb_shape[0]
        if channels not in (1, 3, 4):
            raise ValueError(f"Could not infer RGB channels from shape {rgb_shape}")
        vector_dim = sum(int(torch.tensor(shapes[key]).prod()) for key in self.vector_keys)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.vector_encoder = nn.Sequential(nn.Linear(vector_dim, 128), _activation(activation))
        self.fusion = _mlp(64 * 2 * 2 + 128, (256, 128), activation)
        self.action_head = nn.Linear(128, int(actions_num))
        _initialize(self)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)

    def forward(self, observation: Mapping[str, Tensor]) -> Tensor:
        image = observation[self.rgb_key]
        if image.ndim != 4:
            raise ValueError(f"Batched RGB observation must have four dimensions, received {tuple(image.shape)}")
        if image.shape[-1] in (1, 3, 4):
            image = image.permute(0, 3, 1, 2)
        # Simulation observations are already normalized floats; uint8 is
        # accepted for exported/offline inference without a GPU sync.
        image = image.float().div(255.0) if image.dtype == torch.uint8 else image.float()
        vector = torch.cat([observation[key].float().flatten(start_dim=1) for key in self.vector_keys], dim=-1)
        features = torch.cat((self.image_encoder(image), self.vector_encoder(vector)), dim=-1)
        return self.action_head(self.fusion(features))


if network_builder is not None:
    _BaseNetwork = network_builder.NetworkBuilder.BaseNetwork
    _NetworkBuilder = network_builder.NetworkBuilder
else:
    _BaseNetwork = nn.Module

    class _NetworkBuilder:  # pragma: no cover - only used for clearer errors without RL-Games
        pass


class _TeacherNetwork(_BaseNetwork):
    def __init__(self, params: dict[str, Any], **kwargs: Any) -> None:
        super().__init__()
        actions_num = int(kwargs.pop("actions_num", 1))
        input_shape = _shape(kwargs.pop("input_shape"))
        input_dim = int(torch.tensor(input_shape).prod())
        self.central_value = bool(params.get("central_value", False))
        units = params.get("mlp", {}).get("units", [256, 256, 128])
        activation = params.get("mlp", {}).get("activation", "elu")
        self.trunk = _mlp(input_dim, units, activation)
        output_dim = int(units[-1]) if units else input_dim
        self.mu = nn.Linear(output_dim, actions_num)
        self.value = nn.Linear(output_dim, int(kwargs.pop("value_size", 1)))
        self.sigma = nn.Parameter(torch.zeros(actions_num), requires_grad=True)
        self.fixed_sigma = True
        _initialize(self)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)

    def forward(self, input_dict: Mapping[str, Any]):
        obs = input_dict["obs"]
        if isinstance(obs, Mapping):
            obs = torch.cat([obs[key].float().flatten(start_dim=1) for key in sorted(obs)], dim=-1)
        output = self.trunk(obs.float().flatten(start_dim=1))
        value = self.value(output)
        if self.central_value:
            return value, None
        mu = self.mu(output)
        return mu, mu * 0.0 + self.sigma, value, None

    def is_separate_critic(self) -> bool:
        return False

    def is_rnn(self) -> bool:
        return False

    def get_default_rnn_state(self):
        return None

    def get_value_layer(self):
        return self.value


class BladeSwapTeacherBuilder(_NetworkBuilder):
    """RL-Games builder for the privileged state teacher and central critic."""

    def __init__(self, **_: Any) -> None:
        if network_builder is None:
            raise ImportError("RL-Games is required to construct BladeSwapTeacherBuilder")
        super().__init__()
        self.params: dict[str, Any] = {}

    def load(self, params: dict[str, Any]) -> None:
        self.params = params

    def build(self, _name: str, **kwargs: Any) -> _TeacherNetwork:
        return _TeacherNetwork(self.params, **kwargs)


class _VisionNetwork(_BaseNetwork):
    def __init__(self, params: dict[str, Any], **kwargs: Any) -> None:
        super().__init__()
        actions_num = int(kwargs.pop("actions_num", 1))
        shapes = _shape_dict(kwargs.pop("input_shape"))
        self.central_value = bool(params.get("central_value", False))
        value_size = int(kwargs.pop("value_size", 1))

        if self.central_value:
            input_dim = sum(int(torch.tensor(shape).prod()) for shape in shapes.values())
            units = params.get("mlp", {}).get("units", [256, 256, 128])
            activation = params.get("mlp", {}).get("activation", "elu")
            self.critic = _mlp(input_dim, units, activation)
            output_dim = int(units[-1]) if units else input_dim
            self.value = nn.Linear(output_dim, value_size)
            self.actor = None
            self.sigma = None
        else:
            rgb_key = params.get("rgb_key", "rgb")
            vector_keys = params.get("vector_keys", ["proprio"])
            self.actor = VisionActor(shapes, actions_num, rgb_key, vector_keys, params.get("activation", "elu"))
            self.sigma = nn.Parameter(torch.zeros(actions_num), requires_grad=True)
            self.critic = None
            self.value = nn.Linear(actions_num, value_size)
            checkpoint = params.get("bc_checkpoint")
            if checkpoint:
                payload = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=False)
                state = payload.get("actor_state_dict", payload)
                self.actor.load_state_dict(state, strict=True)
                print(f"[INFO] Loaded behavior-cloned vision actor from {checkpoint}")
        self.fixed_sigma = True

    def forward(self, input_dict: Mapping[str, Any]):
        obs = input_dict["obs"]
        if not isinstance(obs, Mapping):
            raise TypeError("BladeSwapVisionBuilder expected dictionary observations")
        if self.central_value:
            flat = torch.cat([obs[key].float().flatten(start_dim=1) for key in sorted(obs)], dim=-1)
            return self.value(self.critic(flat)), None
        mu = self.actor(obs)
        value = self.value(mu.detach())
        return mu, mu * 0.0 + self.sigma, value, None

    def is_separate_critic(self) -> bool:
        return False

    def is_rnn(self) -> bool:
        return False

    def get_default_rnn_state(self):
        return None

    def get_value_layer(self):
        return self.value


class BladeSwapVisionBuilder(_NetworkBuilder):
    """RL-Games dictionary-observation builder for the deployable vision actor."""

    def __init__(self, **_: Any) -> None:
        if network_builder is None:
            raise ImportError("RL-Games is required to construct BladeSwapVisionBuilder")
        super().__init__()
        self.params: dict[str, Any] = {}

    def load(self, params: dict[str, Any]) -> None:
        self.params = params

    def build(self, _name: str, **kwargs: Any) -> _VisionNetwork:
        return _VisionNetwork(self.params, **kwargs)


def register_rl_games_networks() -> None:
    """Register builders before constructing an RL-Games ``Runner``.

    The NVIDIA Python-3.11 fork used by Isaac Lab 2.3.2 exposes the global
    ``register_network`` function.  Calling this function repeatedly is safe.
    """
    if model_builder is None:
        raise ImportError("RL-Games is not installed; run Isaac Lab's `isaaclab.bat -i rl_games`")
    if not hasattr(model_builder, "register_network"):
        raise RuntimeError("Installed RL-Games does not expose model_builder.register_network")
    model_builder.register_network("blade_swap_teacher", BladeSwapTeacherBuilder)
    model_builder.register_network("blade_swap_vision", BladeSwapVisionBuilder)
