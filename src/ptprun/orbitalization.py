import collections.abc
import logging
import operator
from typing import Optional, Union, cast

import pthelpers as pth
import torch

from .orbit import Orbit

INITIAL_LOGITS_VALUE = 3.0
CLIP_LOGITS_VALUE = 6.0
MULTIPLIER_SIMPLE_LOGITS_VALUE = 0.01

TEMPERATURE = 0.5


logger = logging.getLogger(__name__)


class OrbitBaseLogits(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self) -> torch.Tensor:
        raise NotImplementedError("OrbitsBaseLogits should never be called")

    def probs(self) -> torch.Tensor:
        return torch.sigmoid(self.forward())


class OrbitRawLogits(OrbitBaseLogits):
    def __init__(self, num_logits: int) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.empty((1, num_logits, 1, 1)))
        torch.nn.init.constant_(self.logits, INITIAL_LOGITS_VALUE)

    def forward(self) -> torch.Tensor:
        return self.logits


class OrbitSimpleLogits(OrbitBaseLogits):
    def __init__(self, num_logits: int):
        super().__init__()
        self.num_logits = num_logits
        self.conv_depthwise = torch.nn.Conv2d(
            in_channels=self.num_logits,
            out_channels=self.num_logits,
            groups=self.num_logits,
            kernel_size=3,
            bias=True,
            padding="same",
        )
        self.conv_1x1 = torch.nn.Conv2d(
            in_channels=self.num_logits,
            out_channels=self.num_logits,
            kernel_size=1,
            bias=True,
            padding="same",
        )
        self.logits_clip = CLIP_LOGITS_VALUE
        self.logits_bias = INITIAL_LOGITS_VALUE
        self.logits_multiplier = MULTIPLIER_SIMPLE_LOGITS_VALUE
        self.verbose = False
        x_in = torch.ones(1, self.num_logits, 3, 3)
        self.register_buffer("x_in", x_in, persistent=False)

    def forward(self) -> torch.Tensor:
        x = self.conv_depthwise(self.x_in)
        if self.verbose:
            print(x.shape)
        x = torch.nn.functional.relu(x)
        x = self.conv_1x1(x)
        if self.verbose:
            print(x.shape)
        x = torch.flatten(x, start_dim=2)
        if self.verbose:
            print(x.shape)
        x = torch.mean(x, dim=2)
        if self.verbose:
            print(x.shape)
        x = self.logits_multiplier * x[0] + self.logits_bias
        if self.verbose:
            print(x.shape)
        x = torch.clamp(x, min=-self.logits_clip, max=self.logits_clip)
        if self.verbose:
            print(x.shape)
        x = x.view((1, self.num_logits, 1, 1))
        if self.verbose:
            print(x.shape)
        return x


def _gumbel_difference_like(logits: torch.Tensor) -> torch.Tensor:
    eps = 0.0
    u1 = torch.rand_like(logits)
    u2 = torch.rand_like(logits)
    return -torch.log(torch.log(u2 + eps) / torch.log(u1 + eps) + eps)


def _gumbel_sigmoid(logits: torch.Tensor, t: float) -> torch.Tensor:
    x = logits + _gumbel_difference_like(logits)
    return torch.sigmoid(x / t)


class OrbitMasker(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # For typechecker not to complain
        self.before = torch.tensor(1.0)
        self.after = torch.tensor(1.0)

    def forward(
        self, x: torch.Tensor, logits: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        self.before = x
        res = x * _gumbel_sigmoid(logits, temperature)
        self.after = res
        return res


def _get_num_node_channels(
    node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]
) -> int:
    msg = f"_get_num_node_channes called on {node.op} instead of call_module"
    assert node.op == "call_module", msg
    assert isinstance(node.target, str), "node.target for call_module op is not string"
    if pth.fxfilters.is_linear(node, module_dict):
        res = module_dict[node.target].out_features
        return cast(int, res)
    elif pth.fxfilters.is_conv2d(node, module_dict):
        res = module_dict[node.target].out_channels
        return cast(int, res)
    else:
        msg = f"_get_num_node_channels  module type {type(module_dict[node.target])}"
        msg += " not suppoorted (only linear and conv2d)"
        assert False, msg


def get_num_orbit_channels(gm: torch.fx.GraphModule, o: Orbit) -> int:
    # TODO Replace dict with `getattr` calls
    module_dict = dict(gm.named_modules())
    num_channels = [_get_num_node_channels(n, module_dict) for n in o.sources]
    if len(num_channels) > 1:
        for i, nc in enumerate(num_channels, start=1):
            msg = f"#channels {num_channels[0]} != {nc}, for sink {i}"
            assert num_channels[0] == nc, msg
    return num_channels[0]


def _is_terminal_node(node: torch.fx.Node, orbit: Orbit) -> bool:
    for n in node.users:
        if n in orbit.icns_in_scope:
            return False
    return True


def _is_sink(node: torch.fx.Node, gm: torch.fx.GraphModule) -> bool:
    if node.op == "call_module":
        module = operator.attrgetter(str(node.target))(gm)
        return isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
    return False


def _find_terminal_nodes(o: Orbit, gm: torch.fx.GraphModule) -> set[torch.fx.Node]:
    terminal_nodes = []
    for n in o.icns_in_scope:
        if _is_terminal_node(n, o):
            if _is_sink(n, gm):
                msg = f"sink {n.name} has {len(n.all_input_nodes)} instead of 1"
                assert len(n.all_input_nodes) == 1, msg
                terminal_nodes.append(n.all_input_nodes[0])
            else:
                terminal_nodes.append(n)
    return set(terminal_nodes)


def _get_logits_module_name(color: int) -> str:
    return f"orbit_logits{color}"


def get_orbit_logits_module(
    gm: torch.fx.GraphModule, o: Union[Orbit, int]
) -> OrbitBaseLogits:
    if isinstance(o, Orbit):
        logit_orbits_name = _get_logits_module_name(o.color)
    elif isinstance(o, int):
        logit_orbits_name = _get_logits_module_name(o)
    else:
        raise ValueError(f"Unsupported {type(o)=}, only int and Orbits are supported")
    logits_module = pth.get_module(gm, logit_orbits_name)
    assert isinstance(logits_module, OrbitBaseLogits)
    return logits_module


def get_orbit_logits(gm: torch.fx.GraphModule, o: Union[Orbit, int]) -> torch.Tensor:
    return get_orbit_logits_module(gm, o)()


def orbitalize_in_place(
    gm: torch.fx.GraphModule,
    orbits: list[Orbit],
    device: Optional[torch.device] = None,
    make_logits_module_fn: collections.abc.Callable[
        [int], OrbitBaseLogits
    ] = OrbitSimpleLogits,
) -> None:
    if device is None:
        device = pth.get_device(gm)

    if not isinstance(gm, torch.fx.GraphModule):
        raise ValueError(f"type(gm) should be torch.fx.GraphModule not {type(gm)}")

    if "orbits" in gm.meta:
        raise ValueError("gm is already orbitalized")

    gm.meta["orbits"] = orbits

    for o in reversed(orbits):
        logits_module_name = _get_logits_module_name(o.color)
        logits_num = get_num_orbit_channels(gm, o)
        logits_module = make_logits_module_fn(logits_num)
        logits_module.to(device)
        gm.add_module(logits_module_name, logits_module)
        terminal_nodes = _find_terminal_nodes(o, gm)
        terminal_nodes_names = " ".join(n.name for n in terminal_nodes)

        logger.info(
            f"Orbit O{o.color} found {len(terminal_nodes)} "
            f"terminal nodes: {terminal_nodes_names}"
        )

        with gm.graph.inserting_after(pth.get_first_fxnode(gm)):
            o_logits = gm.graph.call_module(module_name=logits_module_name, args=())

        for i, n in enumerate(terminal_nodes):
            masker_module = OrbitMasker()
            masker_name = f"orbit_masker{o.color}_{i}"
            gm.add_module(masker_name, masker_module)

            # `None` arg in `n_masker` is a placeholder for the old node `n`
            # We want to avoid replacing it with `n_masker`
            # Therefore we substitute `n` for `None` after replacing

            with gm.graph.inserting_after(n):
                n_masker = gm.graph.call_module(
                    module_name=masker_name, args=(None, o_logits, TEMPERATURE)
                )
            n.replace_all_uses_with(n_masker)
            n_masker.update_arg(0, n)

    gm.graph.lint()
    gm.recompile()


def deorbitalize_in_place(
    gm: torch.fx.GraphModule,
) -> None:
    if "orbits" not in gm.meta:
        raise ValueError("gm is not orbitalized")

    # Remove OrbitMaskers

    for node in gm.graph.nodes:
        if node.op == "call_module":
            module = pth.get_module(gm, node.target)
            if isinstance(module, OrbitMasker):
                logger.info(f"Remove OritMasker {node.name}")
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)

    # Remove OrbitLogits

    for node in gm.graph.nodes:
        if node.op == "call_module":
            module = pth.get_module(gm, node.target)
            if isinstance(module, OrbitBaseLogits):
                logger.info(f"Remove OritLogits {node.name}")
                gm.graph.erase_node(node)

    gm.graph.lint()
    gm.delete_all_unused_submodules()
    gm.recompile()
    del gm.meta["orbits"]
