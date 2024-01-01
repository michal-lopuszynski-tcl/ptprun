import logging

import pthelpers as pth
import torch
import torch.fx.passes

from .orbitalization import OrbitMasker

logger = logging.getLogger(__name__)


def _calc_avg_shape(orbit_logits: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.sigmoid(orbit_logits))


def _make_get_logits_for_node(gm, orbits):
    node_to_logits_module = {}

    module_dict = dict(gm.named_modules())
    for o in orbits:
        logits_module_name = f"orbit_logits{o.color}"
        logits_module = module_dict[logits_module_name]

        for node in o.sources:
            node_to_logits_module[node] = logits_module

    def __get_logits(node):
        logits_module = node_to_logits_module.get(node)

        if logits_module is not None:
            return logits_module
        else:
            return lambda: None

    return __get_logits


def _make_unmasked_output_shapes(gm, input_shapes, device):
    sample_input_tensors = (torch.rand(*shape, device=device) for shape in input_shapes)
    torch.fx.passes.shape_prop.ShapeProp(gm).propagate(*sample_input_tensors)
    res = {}
    for n in gm.graph.nodes:
        res[n] = torch.tensor(
            n.meta["tensor_meta"].shape, dtype=torch.float32, device=device
        )
    return res


def _calc_out_shape_conv2d(
    node: torch.fx.Node,
    module_dict,
    logits: torch.Tensor,
    input_shapes: torch.Tensor,
    unmasked_output_shape: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if logits is not None:
        # mask_is_channel_mask = torch.zeros_like(unmasked_output_shape)
        # mask_is_channel_mask[0] = 1.0
        # mask_not_is_channel = -mask_is_channel_mask + 1.0
        # n_channels = _calc_avg_shape(logits)
        # output_shape = unmasked_output_shape * mask_not_is_channel + n_channels * mask_is_channel_mask
        output_shape = unmasked_output_shape.clone()
        output_shape[1] = _calc_avg_shape(logits)
    else:
        output_shape = unmasked_output_shape
    return output_shape


def _calc_out_shape_linear(
    node: torch.fx.Node,
    module_dict,
    logits: torch.Tensor,
    input_shapes: torch.Tensor,
    unmasked_output_shape: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if logits is not None:
        out_features = _calc_avg_shape(logits)
    else:
        linear = module_dict[node.target]
        out_features = torch.tensor(
            linear.out_features, dtype=torch.float32, device=device
        )
    output_shape = input_shapes[0].clone()
    output_shape[-1] = out_features
    return output_shape


def _calc_out_shape_flatten(
    node: torch.fx.Node,
    module_dict,
    logits: torch.Tensor,
    input_shapes: torch.Tensor,
    unmasked_output_shape: torch.Tensor,
    device: torch.device,
):
    input_shape = input_shapes[0]
    kwargs_default = {"end_dim": -1}
    kwargs_args = dict(zip(["input", "start_dim", "end_dim"], node.args))
    kwargs_tot = kwargs_default | node.kwargs | kwargs_args
    start_dim = kwargs_tot["start_dim"]
    end_dim = kwargs_tot["end_dim"] % len(input_shape) + 1
    flatten_shape = torch.prod(input_shape[start_dim:end_dim]).unsqueeze(dim=0)
    res = torch.cat([input_shape[0:start_dim], flatten_shape, input_shape[end_dim:]])
    return res


def _calc_out_shape_binary_op(
    node: torch.fx.Node,
    module_dict,
    logits: torch.Tensor,
    input_shapes: torch.Tensor,
    unmasked_output_shape: torch.Tensor,
    device: torch.device,
):
    input_shape1 = input_shapes[0]
    input_shape2 = input_shapes[1]
    max_dim = max(len(input_shape1), len(input_shape2))
    input_shape1 = torch.nn.functional.pad(
        input_shape1, (max_dim - len(input_shape1), 0)
    )
    input_shape2 = torch.nn.functional.pad(
        input_shape2, (max_dim - len(input_shape2), 0)
    )
    return torch.maximum(input_shape1, input_shape2)


def _calc_out_shape_copy_input_channels(
    node: torch.fx.Node,
    module_dict,
    logits: torch.Tensor,
    input_shapes: torch.Tensor,
    unmasked_output_shape: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    # Shape prop
    output_shape = unmasked_output_shape.clone()
    output_shape[1] = input_shapes[0][1]
    return output_shape


def _calc_out_shape_copy_input_shape(
    node: torch.fx.Node,
    module_dict,
    logits: torch.Tensor,
    input_shapes: torch.Tensor,
    unmasked_output_shape: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return input_shapes[0]


def _calc_fpops_zero(
    node: torch.fx.Node,
    module_dict,
    input_shapes: list[torch.Tensor],
    output_shape: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(0.0, dtype=torch.float32, device=device)


def _is_orbit_masker(node, module_dict):
    return node.op == "call_module" and isinstance(
        module_dict[node.target], OrbitMasker
    )


def _is_pool2d_nonadaptive(node, module_dict):
    return pth.fxfilters.is_pool2d_avg(
        node, module_dict
    ) or pth.fxfilters.is_pool2d_max(node, module_dict)


def _is_pool2d_adaptive(node, module_dict):
    return pth.fxfilters.is_pool2d_adaptive_avg(
        node, module_dict
    ) or pth.fxfilters.is_pool2d_adaptive_max(node, module_dict)


def _add_device_to_args(calc_fpops_fn):
    def __calc_fpops_wrapped(
        node: torch.fx.Node,
        module_dict,
        input_shapes: list[torch.Tensor],
        output_shape: torch.Tensor,
        device: torch.device,
    ):
        return calc_fpops_fn(node, module_dict, input_shapes, output_shape)

    return __calc_fpops_wrapped


def make_default_calc_fpops_prop_startegy():
    return [
        (
            pth.fxfilters.is_conv2d_plain,
            _calc_out_shape_conv2d,
            _add_device_to_args(pth.stats.calc_fpops_conv2d_plain),
        ),
        (
            pth.fxfilters.is_conv2d_dwise,
            _calc_out_shape_copy_input_channels,
            _add_device_to_args(pth.stats.calc_fpops_conv2d_dwise),
        ),
        (
            pth.fxfilters.is_linear,
            _calc_out_shape_linear,
            _add_device_to_args(pth.stats.calc_fpops_linear),
        ),
        (
            _is_pool2d_nonadaptive,
            _calc_out_shape_copy_input_channels,
            _add_device_to_args(pth.stats.calc_fpops_pool2d),
        ),
        (
            _is_pool2d_adaptive,
            _calc_out_shape_copy_input_channels,
            _add_device_to_args(pth.stats.calc_fpops_pool2d_adaptive),
        ),
        (
            pth.fxfilters.is_batchnorm2d,
            _calc_out_shape_copy_input_channels,
            _add_device_to_args(pth.stats.calc_fpops_batchnorm2d),
        ),
        (
            pth.fxfilters.is_activation,
            _calc_out_shape_copy_input_channels,
            _calc_fpops_zero,
        ),
        (
            _is_orbit_masker,
            _calc_out_shape_copy_input_shape,
            _calc_fpops_zero,
        ),
        (
            pth.fxfilters.is_add,
            _calc_out_shape_binary_op,
            _add_device_to_args(pth.stats.calc_fpops_add_mul),
        ),
        (
            pth.fxfilters.is_mul,
            _calc_out_shape_binary_op,
            _add_device_to_args(pth.stats.calc_fpops_add_mul),
        ),
        (
            pth.fxfilters.is_flatten,
            _calc_out_shape_flatten,
            _calc_fpops_zero,
        ),
        (
            pth.fxfilters.is_dropout,
            _calc_out_shape_copy_input_channels,
            _calc_fpops_zero,
        ),
    ]


def _make_split_fn(startegy):
    def __split_fn(node, module_dict):
        for match_fn, _, _ in startegy:
            if match_fn(node, module_dict):
                return match_fn.__name__
        else:
            return "other"

    return __split_fn


def check_calc_fpops_strategy(gm: torch.fx.GraphModule, strategy):
    split_fn = _make_split_fn(strategy)
    return pth.stats.calc_fx_ops_per_nn_op(gm, split_fn)


def _make_calc_fpops_helper_kwargs(
    *,
    orbitalized_gm: torch.fx,
    input_shapes: tuple[int, ...],
    device: torch.device,
    make_calc_fpops_startegy_fn,
    unit: str = "kmapps",
):
    kwargs = {"device": device}
    orbits = orbitalized_gm.meta["orbits"]
    kwargs["get_logits_for_node"] = _make_get_logits_for_node(orbitalized_gm, orbits)
    kwargs["fpops_prop_strategy"] = make_calc_fpops_startegy_fn()
    kwargs["divisor"] = pth.stats.get_fpops_divisor(unit, input_shapes)
    module_dict = dict(orbitalized_gm.named_modules())
    kwargs["module_dict"] = module_dict
    unmasked_output_shapes = _make_unmasked_output_shapes(
        orbitalized_gm, input_shapes, device
    )
    kwargs["unmasked_output_shapes"] = unmasked_output_shapes

    fixed_shapes = {}

    i = 0
    for node in orbitalized_gm.graph.nodes:
        if pth.fxfilters.is_input(node, module_dict):
            fixed_shapes[node] = torch.tensor(
                input_shapes[i], dtype=torch.float32, device=device
            )
        elif pth.fxfilters.is_without_args(node, module_dict):
            fixed_shapes[node] = unmasked_output_shapes[node]
    kwargs["fixed_shapes"] = fixed_shapes
    return kwargs


def _calc_fpops_helper(
    *,
    orbitalized_gm,
    fixed_shapes,
    module_dict,
    fpops_prop_strategy,
    get_logits_for_node,
    unmasked_output_shapes,
    device,
    divisor,
):
    output_shapes = fixed_shapes.copy()
    fpops_tot = torch.tensor(0.0, dtype=torch.float32, device=device)

    for node in orbitalized_gm.graph.nodes:
        for filter_fn, calc_shape_fn, calc_fpops_fn in fpops_prop_strategy:
            if pth.fxfilters.is_without_args(
                node, module_dict
            ) or pth.fxfilters.is_output(node, module_dict):
                break
            elif filter_fn(node, module_dict):
                op_input_shapes = [output_shapes[nn] for nn in node.all_input_nodes]
                logits = get_logits_for_node(node)()
                op_output_shape = calc_shape_fn(
                    node=node,
                    module_dict=module_dict,
                    logits=logits,
                    input_shapes=op_input_shapes,
                    unmasked_output_shape=unmasked_output_shapes[node],
                    device=device,
                )
                # if (
                #     pth.fxfilters.is_linear(node, module_dict)
                #     or pth.fxfilters.is_conv2d(node, module_dict)
                #     or _is_pool2d_nonadaptive(node, module_dict)
                # ):
                #     if logits is not None:
                #         logger.info(f"Fpo: {node.name} logits={logits[0][0]}")
                #     else:
                #         logger.info(f"Fpo: {node.name} logits=None")
                #     logger.info(
                #         f"Fpo: {node.name}, in_channels={op_input_shapes[0][1]}"
                #         f" out_channels={op_output_shape[1]}"
                #     )
                fpops = calc_fpops_fn(
                    node, module_dict, op_input_shapes, op_output_shape, device
                )
                output_shapes[node] = op_output_shape
                fpops_tot += fpops
                break
        else:
            # TODO Add some default handling here
            fx_node_name = pth.get_fxnode_name(node, module_dict)
            msg = f"Unsupported operation for node {node.name} ({fx_node_name})"
            raise ValueError(msg)

    return fpops_tot / divisor


def calc_fpops(
    *,
    orbitalized_gm: torch.fx,
    input_shapes: tuple[int, ...],
    device: torch.device,
    make_calc_fpops_startegy_fn=make_default_calc_fpops_prop_startegy,
    unit: str = "kmapps",
) -> torch.Tensor:
    kwargs = _make_calc_fpops_helper_kwargs(
        orbitalized_gm=orbitalized_gm,
        input_shapes=input_shapes,
        device=device,
        make_calc_fpops_startegy_fn=make_calc_fpops_startegy_fn,
        unit=unit,
    )
    return _calc_fpops_helper(orbitalized_gm=orbitalized_gm, **kwargs)


def make_calc_fpops_fn(
    *,
    orbitalized_gm: torch.fx,
    input_shapes: tuple[int, ...],
    device: torch.device,
    make_calc_fpops_startegy_fn=make_default_calc_fpops_prop_startegy,
    unit: str = "kmapps",
):
    kwargs = _make_calc_fpops_helper_kwargs(
        orbitalized_gm=orbitalized_gm,
        input_shapes=input_shapes,
        device=device,
        make_calc_fpops_startegy_fn=make_calc_fpops_startegy_fn,
        unit=unit,
    )

    def __calc_fpops(model):
        msg = "You can use fpops_fn only for the model it was built for"
        assert model is orbitalized_gm, msg
        return _calc_fpops_helper(orbitalized_gm=model, **kwargs)

    return __calc_fpops


def make_loss_fpops_fn(
    target_fpops,
    orbitalized_gm,
    input_shapes,
    device,
    make_calc_fpops_strategy_fn=make_default_calc_fpops_prop_startegy,
    unit: str = "kmapps",
):
    kwargs = _make_calc_fpops_helper_kwargs(
        orbitalized_gm=orbitalized_gm,
        input_shapes=input_shapes,
        device=device,
        make_calc_fpops_startegy_fn=make_calc_fpops_strategy_fn,
        unit=unit,
    )

    def __calc_loss_fpops(model):
        msg = "You can use fpops_fn only for the model it was built for"
        assert model is orbitalized_gm, msg
        fpops = _calc_fpops_helper(orbitalized_gm=model, **kwargs)
        return torch.nn.functional.relu(fpops / target_fpops - 1.0)

    return __calc_loss_fpops


def calc_logits_entropy(
    logits: torch.Tensor, t: torch.Tensor, epsilon: float = 1.0e-2
) -> torch.Tensor:
    ## TODO Check the epsilon value, it seems quite large
    probs_channel_on = torch.sigmoid(logits / t)
    probs_channel_off = -probs_channel_on + 1.0
    entropy_per_channel = (
        -torch.log(probs_channel_on) * probs_channel_on
        - torch.log(probs_channel_off) * probs_channel_off
    )
    return torch.maximum(entropy_per_channel.mean(), torch.tensor(epsilon))


def calc_loss_logits(orbitalized_gm: torch.fx.GraphModule) -> torch.Tensor:
    tot_loss = None
    orbits = orbitalized_gm.meta["orbits"]
    module_dict = dict(orbitalized_gm.named_modules())

    for o in orbits:
        logits_name = f"orbit_logits{o.color}"
        logits = module_dict[logits_name]()
        # TODO What about temperature?
        t = 1.0
        logits_entropy = calc_logits_entropy(logits, t)
        if tot_loss is None:
            tot_loss = logits_entropy
        else:
            tot_loss += logits_entropy
    return tot_loss / len(orbits)


def calc_noise_to_signal_ratio(
    outputs: torch.Tensor, targets: torch.Tensor, epsilon: float = 1.0e-6
) -> torch.Tensor:
    dim = tuple([0] + list(range(2, outputs.ndim)))
    variance_per_channel = torch.var(targets, dim=dim, keepdim=False, unbiased=False)
    diff2_per_channel = torch.mean((outputs - targets) ** 2, dim=dim, keepdim=False)
    nsr = diff2_per_channel / (variance_per_channel + epsilon)
    return nsr.mean()


def calc_loss_noise_to_signal_ratio(
    orbitalized_gm: torch.fx.GraphModule,
) -> torch.Tensor:
    tot_nsr = None
    n = 0

    for mod in orbitalized_gm.modules():
        if isinstance(mod, OrbitMasker):
            nsr = calc_noise_to_signal_ratio(mod.after, mod.before)
            n += 1
            if tot_nsr is None:
                tot_nsr = nsr
            else:
                tot_nsr += nsr

    return tot_nsr / n
