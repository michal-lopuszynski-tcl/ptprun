import collections
import logging

import pthelpers as pth
import torch

from . import orbitalization
from .orbit import Orbit

logger = logging.getLogger(__name__)

_NO_PRUN_FLAG = -1.0


def _propagate_mask_single_input(node, module_dict, predecessor_masks):
    return predecessor_masks[0]


def _propagate_mask_binary_op(node, module_dcit, predecessor_masks):
    if len(predecessor_masks) > 1:
        for pm in predecessor_masks[1:]:
            assert torch.all(pm == predecessor_masks[0]).item()
    return predecessor_masks[0]


def _propagate_mask_cat_channels(node, module_dcit, predecessor_masks):
    return torch.cat(predecessor_masks)


def _propagate_mask_conv2d(node, module_dict, predecessor_mask):
    return torch.ones(module_dict[node.target].out_channels, dtype=torch.bool)


def _propagate_mask_linear(node, module_dict, predecesor_maks):
    return torch.ones(module_dict[node.target].out_features, dtype=torch.bool)


def _propagate_mask_no_prun(node, module_dict, predecessor_mask):
    return torch.tensor(_NO_PRUN_FLAG)


def _is_any(filter_fns, node, module_dict):
    for filter_fn in filter_fns:
        if filter_fn(node, module_dict):
            return True
    return False


def _is_simple_op(node, module_dict):
    simple_op_filter_fns = [
        pth.fxfilters.is_activation,
        pth.fxfilters.is_batchnorm2d,
        pth.fxfilters.is_pool2d_avg,
        pth.fxfilters.is_pool2d_max,
        pth.fxfilters.is_pool2d_adaptive_avg,
        pth.fxfilters.is_pool2d_adaptive_max,
    ]
    return _is_any(simple_op_filter_fns, node, module_dict)


def make_default_propagate_masks_strategy():
    # Safe substitute for constant list declaration, which Python does not support
    return [
        (_is_simple_op, _propagate_mask_single_input),
        (pth.fxfilters.is_binary_op, _propagate_mask_binary_op),
        (pth.fxfilters.is_cat_channels, _propagate_mask_cat_channels),
        (pth.fxfilters.is_conv2d_plain, _propagate_mask_conv2d),
        (pth.fxfilters.is_linear, _propagate_mask_linear),
        (pth.fxfilters.is_flatten, _propagate_mask_no_prun),
        (pth.fxfilters.is_output, _propagate_mask_no_prun),
    ]


def _propagate_masks(
    graph_module: torch.fx.graph_module.GraphModule,
    node_to_mask: dict[torch.fx.node.Node, torch.Tensor],
    propagate_masks_strategy,
) -> dict[torch.fx.node.Node, torch.Tensor]:
    module_dict = dict(graph_module.named_modules())

    full_node_to_mask = {}
    for node in graph_module.graph.nodes:
        if node in node_to_mask:
            full_node_to_mask[node] = node_to_mask[node]
        else:
            predecessor_masks = [full_node_to_mask[np] for np in node.all_input_nodes]
            for filter_fn, propagate_mask_fn in propagate_masks_strategy:
                if filter_fn(node, module_dict):
                    full_node_to_mask[node] = propagate_mask_fn(
                        node, module_dict, predecessor_masks
                    )
                    break
            else:
                raise ValueError(f"Mask prop for node {node} not supported")
    return full_node_to_mask


def _is_non_distilled(node, module_dict):
    non_distilled_filter_fns = [pth.fxfilters.is_without_params]
    return _is_any(non_distilled_filter_fns, node, module_dict)


def _make_batchnorm2d_config_dict(node_module):
    node_params = ["num_features", "eps", "momentum", "affine", "track_running_stats"]
    config_dict = {k: vars(node_module)[k] for k in node_params}
    return config_dict


def _make_distilled_batchnorm2d(node_module, output_mask, predecessor_masks):
    assert len(predecessor_masks) == 1, f"{len(predecessor_masks)=}"
    n_features_in = int(torch.sum(predecessor_masks[0]).item())
    assert n_features_in == int(torch.sum(output_mask).item())

    config_dict = _make_batchnorm2d_config_dict(node_module)
    logger.info(f"Original batchnorm2d {config_dict}")
    config_dict["num_features"] = n_features_in
    logger.info(f"Creating new batchnorm2d {config_dict}")
    new_node_module = torch.nn.BatchNorm2d(**config_dict)

    state_dict = node_module.state_dict()
    device = state_dict["weight"].device
    new_state_dict = collections.OrderedDict()

    for k, v in state_dict.items():
        if k != "num_batches_tracked":
            new_state_dict[k] = v[predecessor_masks[0]]
    new_node_module.load_state_dict(new_state_dict)
    new_node_module.to(device)
    return new_node_module


def _make_conv2d_config_dict(node_module):
    node_params = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
    ]
    config_dict = {k: vars(node_module)[k] for k in node_params}
    config_dict["bias"] = node_module.bias is not None
    return config_dict


def _make_distilled_conv2d_plain(node_module, output_mask, predecessor_masks):
    config_dict = _make_conv2d_config_dict(node_module)
    logger.info(f"Original conv2d {config_dict}")

    config_dict["out_channels"] = int(sum(output_mask).item())
    config_dict["in_channels"] = int(sum(predecessor_masks[0]).item())
    logger.info(f"Creating new conv2d {config_dict}")
    new_node_module = torch.nn.Conv2d(**config_dict)

    state_dict = node_module.state_dict()
    device = state_dict["weight"].device
    weight = state_dict["weight"]
    weight_new = weight[output_mask, :, :, :]
    weight_new = weight_new[:, predecessor_masks[0], :, :]

    new_state_dict = collections.OrderedDict()
    new_state_dict["weight"] = weight_new

    if node_module.bias is not None:
        bias = state_dict["bias"]
        bias_new = bias[output_mask]
        new_state_dict["bias"] = bias_new

    new_node_module.load_state_dict(new_state_dict)
    new_node_module.to(device)
    return new_node_module


def make_default_distill_strategy():
    # Safe substitute for constant list declaration, which Python does not support
    return [
        (pth.fxfilters.is_conv2d_plain, _make_distilled_conv2d_plain),
        (pth.fxfilters.is_batchnorm2d, _make_distilled_batchnorm2d),
        # This is to explicitly mark supported modules
        (_is_non_distilled, None),
    ]


def _distill_node_in_place(
    graph_module: torch.fx.graph_module.GraphModule,
    node: torch.fx.Node,
    output_mask: torch.Tensor,
    predecessor_masks: list[torch.Tensor],
    distill_strategy,
) -> None:
    module_dict = dict(graph_module.named_modules())
    for filter_fn, make_distilled_module_fn in distill_strategy:
        if filter_fn(node, module_dict):
            if make_distilled_module_fn is not None:
                node_module = pth.get_module(graph_module, node.target)
                new_node_module = make_distilled_module_fn(
                    node_module, output_mask, predecessor_masks
                )
                pth.replace_fxsubmodule(graph_module, node, new_node_module)
            return

    raise ValueError(f"Distilling for {node} not implemented")


def _contains_distillable_mask(
    predecessor_masks: list[torch.Tensor], output_mask: torch.Tensor
) -> bool:
    output_prun = output_mask.numel() != 1 or output_mask.item() != _NO_PRUN_FLAG
    if output_prun:
        output_non_trivial = output_mask.numel() != torch.sum(output_mask)
        if output_non_trivial:
            return True

    for mask in predecessor_masks:
        mask_prun = mask.numel() != 1 or mask.item() != _NO_PRUN_FLAG
        if mask_prun:
            mask_non_trivial = mask.numel() != torch.sum(mask)
            if mask_non_trivial:
                return True
    return False


def _log_node_to_mask(node_to_mask):
    for k, v in node_to_mask.items():
        n_tot = v.numel()
        if v.numel() == 1 and v.item() == _NO_PRUN_FLAG:
            n_tot = -1
            n_distilled = -1
        else:
            n_distilled = torch.sum(v).item()

        logger.info(f"{k.name:25s}: {n_tot:3d} -> {n_distilled:3d}")


def distill_module_in_place(
    graph_module: torch.fx.graph_module.GraphModule,
    node_to_mask: dict[torch.fx.node.Node, torch.Tensor],
    make_propagate_masks_strategy_fn=make_default_propagate_masks_strategy,
    make_distill_strategy_fn=make_default_distill_strategy,
) -> None:
    propagate_masks_strategy = make_propagate_masks_strategy_fn()
    distill_strategy = make_distill_strategy_fn()

    full_node_to_mask = _propagate_masks(
        graph_module, node_to_mask, propagate_masks_strategy
    )
    _log_node_to_mask(full_node_to_mask)
    for node in full_node_to_mask.keys():
        predecessor_masks = [
            full_node_to_mask[node_pred] for node_pred in node.all_input_nodes
        ]
        output_mask = full_node_to_mask[node]
        if _contains_distillable_mask(predecessor_masks, output_mask):
            logger.info(f"Distilling {node.name}")
            _distill_node_in_place(
                graph_module=graph_module,
                node=node,
                output_mask=output_mask,
                predecessor_masks=predecessor_masks,
                distill_strategy=distill_strategy,
            )
        else:
            logger.info(f"Not distilling {node.name}, only trivial masks found")

    graph_module.recompile()


def get_orbits_distilling_mask(
    graph_module: torch.fx.graph_module.GraphModule,
    orbits: list[Orbit],
    get_orbit_mask_fn,
) -> dict[torch.fx.Node, torch.Tensor]:
    node_to_mask = {pth.get_fxnode(graph_module, 0): torch.ones(3, dtype=torch.bool)}

    for o in orbits:
        mask = get_orbit_mask_fn(graph_module, o)
        n_all = sum(torch.ones_like(mask))
        n_pru = sum(mask)
        logger.info(f"orbit color={o.color}: n_channels {n_all} -> {n_pru}")
        for n in o.sources + list(o.non_border):
            node_to_mask[n] = mask

    return node_to_mask


def distill_orbitalized_module_in_place(
    graph_module: torch.fx.graph_module.GraphModule,
    make_propagate_masks_strategy_fn=make_default_propagate_masks_strategy,
    make_distill_strategy_fn=make_default_distill_strategy,
) -> None:
    orbits = graph_module.meta["orbits"]

    # Get orbits distilling mask
    node_to_mask = {pth.get_fxnode(graph_module, 0): torch.ones(3, dtype=torch.bool)}
    for o in orbits:
        logits = orbitalization.get_orbit_logits(graph_module, o).flatten()
        mask = logits > 0.0
        for n in o.sources + list(o.non_border):
            node_to_mask[n] = mask

    # Deorbitalize
    orbitalization.deorbitalize_in_place(graph_module)

    # TODO Remove pruned paths

    # Distill
    distill_module_in_place(
        graph_module,
        node_to_mask,
        make_propagate_masks_strategy_fn,
        make_distill_strategy_fn,
    )
