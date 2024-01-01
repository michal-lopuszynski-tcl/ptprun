import contextlib
import logging

import pytest
import torch

import ptprun

import sample_models

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_msg_fn(val, expected_val):
    def _make_msg(msg):
        msg += f"\nValues: got={val.item()} expected={expected_val.item()}"
        return msg

    return _make_msg


def calc_alexnet_fpops_loss(
    model: torch.fx.GraphModule, return_fpops: bool
) -> torch.Tensor:
    target_fpops = 30.0
    factor = 224.0 * 224.0 * 1000.0

    # Conv1
    in_channels = 3.0
    out_channels = torch.sum(model.orbit_logits0.probs())
    out_channels_ = 96.0
    out_h, out_w = 54.0, 54.0
    k_h, k_w = 11.0, 11.0
    conv1_ops = 2.0 * out_h * out_w * k_h * k_w * in_channels * out_channels
    logger.info(f"Ref: conv1, {in_channels=} {out_channels=} {out_channels_=}")

    # Pool1
    # Out chanannels remain unchanged
    out_h, out_w = 26, 26
    k_h, k_w = 3, 3
    pool1_ops = out_h * out_w * out_channels * k_h * k_w
    logger.info(f"Ref: pool1_ops, {in_channels=} {out_channels=} {out_channels_=}")

    # Conv2
    in_channels = out_channels
    out_channels = torch.sum(model.orbit_logits1.probs())
    out_channels_ = 256.0
    out_h, out_w = 26.0, 26.0
    k_h, k_w = 5.0, 5.0
    conv2_ops = 2.0 * out_h * out_w * k_h * k_w * in_channels * out_channels
    logger.info(f"Ref: conv2, {in_channels=} {out_channels=} {out_channels_=}")

    # Pool2
    # Out chanannels remain unchanged
    out_h, out_w = 12, 12
    k_h, k_w = 3, 3
    pool2_ops = out_h * out_w * out_channels * k_h * k_w
    logger.info(f"Ref: pool2_ops, {in_channels=} {out_channels=} {out_channels_=}")

    # Conv3
    in_channels = out_channels
    out_channels = torch.sum(model.orbit_logits2.probs())
    out_channels_ = 384.0
    out_h, out_w = 12.0, 12.0
    k_h, k_w = 3.0, 3.0
    conv3_ops = 2.0 * out_h * out_w * k_h * k_w * in_channels * out_channels
    logger.info(f"Ref: conv3, {in_channels=} {out_channels=} {out_channels_=}")

    # Conv4
    in_channels = out_channels
    out_channels = torch.sum(model.orbit_logits3.probs())
    out_channels_ = 384.0
    out_h, out_w = 12.0, 12.0
    k_h, k_w = 3.0, 3.0
    conv4_ops = 2.0 * out_h * out_w * k_h * k_w * in_channels * out_channels
    logger.info(f"Ref: conv4 {in_channels=} {out_channels=} {out_channels_=}")

    # Conv5
    in_channels = out_channels
    out_channels = torch.tensor(256.0)
    out_channels_ = 256.0
    out_h, out_w = 12.0, 12.0
    k_h, k_w = 3.0, 3.0
    conv5_ops = 2.0 * out_h * out_w * k_h * k_w * in_channels * out_channels
    logger.info(f"Ref: conv5 {in_channels=} {out_channels=} {out_channels_=}")

    # Pool3
    # Out chanannels remain unchanged
    out_h, out_w = 5, 5
    k_h, k_w = 3, 3
    pool3_ops = out_h * out_w * out_channels * k_h * k_w
    logger.info(f"Ref: pool3_ops, {in_channels=} {out_channels=} {out_channels_=}")

    # L1
    in_channels = torch.tensor(6400)

    out_channels = torch.sum(model.orbit_logits4.probs())
    out_channels_ = torch.tensor(4096)
    l1_ops = (2.0 * in_channels + 1.0) * out_channels

    # L2
    in_channels = out_channels
    out_channels = torch.tensor(10)
    l2_ops = (2.0 * in_channels + 1.0) * out_channels

    fpops = (
        conv1_ops
        + conv2_ops
        + conv3_ops
        + conv4_ops
        + conv5_ops
        + pool1_ops
        + pool2_ops
        + pool3_ops
        + l1_ops
        + l2_ops
    )
    fpops /= torch.tensor(factor)
    if return_fpops:
        return fpops
    else:
        return torch.nn.functional.relu(fpops / target_fpops - 1.0)


@contextlib.contextmanager
def fixed_logits(orbitalized_gm: torch.fx.GraphModule, logits_value: float) -> None:
    module_dict = dict(orbitalized_gm.named_modules())

    logits_m_pars = {}

    for m_name, m in module_dict.items():
        if m_name.startswith("orbit_logits") and "." not in m_name:
            logits_m_pars[m_name] = (m.logits_clip, m.logits_bias, m.logits_multiplier)
            m.logits_clip = 1.1 * abs(logits_value)
            m.logits_bias = logits_value
            m.logits_multiplier = 0.0

    yield None

    for m_name, m_pars in logits_m_pars.items():
        m = module_dict[m_name]
        (m.logits_clip, m.logits_bias, m.logits_multiplier) = m_pars


@pytest.mark.parametrize("logits_value", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 100])
def test_make_calc_flops_fn_alexnet(logits_value):
    input_shape = (1, 3, 224, 224)
    model = sample_models.AlexNet(num_classes=10)
    tr_model = torch.fx.symbolic_trace(model)
    device = torch.device("cpu")
    orbits = ptprun.extraction.extract_orbits(tr_model)
    ptprun.orbitalization.orbitalize_in_place(tr_model, orbits, device)

    calc_fpops_fn = ptprun.losses.make_calc_fpops_fn(
        orbitalized_gm=tr_model,
        input_shapes=(input_shape,),
        device=device,
        unit="kmapps",
    )

    with fixed_logits(tr_model, logits_value):
        fpops_expected = calc_alexnet_fpops_loss(model=tr_model, return_fpops=True)
        fpops = calc_fpops_fn(tr_model)
        torch.testing.assert_close(
            fpops, fpops_expected, msg=make_msg_fn(fpops, fpops_expected)
        )


TEST_DATA_ALEXNET = [
    (-1.0, 5.6234145),
    (0.0, 12.933848),
    (1.0, 23.367552),
    (2.0, 31.797026),
    (3.0, 36.302765),
    (4.0, 38.23797),
    (5.0, 38.993797),
    (100.0, 39.445244),
]


@pytest.mark.parametrize("logits_value, kmapps", TEST_DATA_ALEXNET)
def test_make_calc_flops_fn_alexnet_node_api(logits_value, kmapps):
    input_shape = (1, 3, 224, 224)
    model = sample_models.alexnet(num_classes=10)
    tr_model = torch.fx.symbolic_trace(model)
    device = torch.device("cpu")
    orbits = ptprun.extraction.extract_orbits(tr_model)
    # For now reference data does not include l1 pruning gains
    orbits = [o for o in orbits if not ptprun.orbit.is_orbit_with_node_name(o, "l1")]
    ptprun.orbitalization.orbitalize_in_place(tr_model, orbits, device)

    calc_fpops_fn = ptprun.losses.make_calc_fpops_fn(
        orbitalized_gm=tr_model,
        input_shapes=(input_shape,),
        device=device,
        unit="kmapps",
    )

    logger.info(f"{len(orbits)=}")
    with fixed_logits(tr_model, logits_value):
        fpops_expected = torch.tensor(kmapps)
        fpops_1 = calc_fpops_fn(tr_model)
        torch.testing.assert_close(
            fpops_1, fpops_expected, msg=make_msg_fn(fpops_1, fpops_expected)
        )
