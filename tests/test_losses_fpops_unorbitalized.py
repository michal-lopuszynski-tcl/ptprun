import numpy as np
import pthelpers as pth
import pytest
import torch

import ptprun.losses

import sample_models

DTYPE = torch.float32


def _make_msg_fn(val, expected_val):
    def _make_msg(msg):
        msg += f"\nValues: got={val.item()} expected={expected_val.item()}"
        return msg

    return _make_msg


def _assert_close(val, expected_val):
    torch.testing.assert_close(val, expected_val, msg=_make_msg_fn(val, expected_val))


class SimpleModule1(torch.nn.Module):
    def __init__(self, bias1, bias2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            kernel_size=(3, 3),
            out_channels=8,
            padding="same",
            bias=bias1,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=8,
            kernel_size=(3, 3),
            out_channels=3,
            padding="same",
            bias=bias2,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = x + x2
        x4 = torch.flatten(x3, start_dim=1)
        return x4


class SimpleModule2(torch.nn.Module):
    def __init__(self, bias1, bias2):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=16, out_features=8, bias=bias1)
        self.linear2 = torch.nn.Linear(in_features=8, out_features=16, bias=bias2)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        x3 = x + x2
        x4 = torch.flatten(x3, start_dim=1)
        return x4


class SimpleModule3(torch.nn.Module):
    def __init__(self, bias1, bias2):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=32, out_features=11, bias=bias1)
        self.linear2 = torch.nn.Linear(in_features=11, out_features=8, bias=bias2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.flatten(x, start_dim=1)
        return x


TEST_COMPARE_LOSSES_AGAINST_STATS = [
    ["simplemodule1a", SimpleModule1(bias1=False, bias2=False), (1, 3, 224, 224)],
    ["simplemodule1b", SimpleModule1(bias1=True, bias2=False), (1, 3, 224, 224)],
    ["simplemodule1c", SimpleModule1(bias1=False, bias2=True), (1, 3, 224, 224)],
    ["simplemodule1d", SimpleModule1(bias1=True, bias2=True), (1, 3, 224, 224)],
    ["simplemodule2a", SimpleModule2(bias1=False, bias2=False), (1, 5, 7, 16)],
    ["simplemodule2b", SimpleModule2(bias1=False, bias2=True), (1, 5, 7, 16)],
    ["simplemodule2c", SimpleModule2(bias1=True, bias2=False), (1, 5, 7, 16)],
    ["simplemodule2d", SimpleModule2(bias1=True, bias2=True), (1, 5, 7, 16)],
    ["alexnet", sample_models.alexnet(num_classes=10), (1, 3, 224, 224)],
    ["resnet18", sample_models.resnet18(num_classes=10), (1, 3, 224, 224)],
]


@pytest.mark.parametrize(
    "model_name, model, input_shape", TEST_COMPARE_LOSSES_AGAINST_STATS
)
def test_calc_fpops_against_stats(model_name, model, input_shape):
    tr_model = torch.fx.symbolic_trace(model)
    tr_model.meta["orbits"] = []
    device = torch.device("cpu")
    fpops_expected = pth.stats.calc_fpops(
        tr_model, input_shapes=(input_shape,), unit="flops"
    )
    fpops_expected = torch.tensor(fpops_expected, dtype=DTYPE)
    fpops = ptprun.losses.calc_fpops(
        orbitalized_gm=tr_model,
        input_shapes=(input_shape,),
        device=device,
        unit="flops",
    )
    _assert_close(fpops_expected, fpops)


@pytest.mark.parametrize(
    "model_name, model, input_shape", TEST_COMPARE_LOSSES_AGAINST_STATS
)
def test_make_calc_fpops_fn_against_stats(model_name, model, input_shape):
    tr_model = torch.fx.symbolic_trace(model)
    tr_model.meta["orbits"] = []
    device = torch.device("cpu")
    fpops_expected = pth.stats.calc_fpops(
        tr_model, input_shapes=(input_shape,), unit="kmapps"
    )
    fpops_expected = torch.tensor(fpops_expected, dtype=DTYPE)

    calc_fpops_fn = ptprun.losses.make_calc_fpops_fn(
        orbitalized_gm=tr_model,
        input_shapes=(input_shape,),
        device=device,
        unit="kmapps",
    )

    # Make sure it is truly stateless, hence computing the same loss twice
    fpops1 = calc_fpops_fn(tr_model)
    fpops2 = calc_fpops_fn(tr_model)
    _assert_close(fpops1, fpops_expected)
    _assert_close(fpops2, fpops_expected)


@pytest.mark.parametrize(
    "model_name, model, input_shape", TEST_COMPARE_LOSSES_AGAINST_STATS
)
def test_make_loss_fpops_fn_against_stats(model_name, model, input_shape):
    device = torch.device("cpu")
    tr_model = torch.fx.symbolic_trace(model)
    tr_model.meta["orbits"] = []
    true_fpops = pth.stats.calc_fpops(
        m=tr_model, input_shapes=(input_shape,), unit="kmapps"
    )

    fpops_loss_fn = ptprun.losses.make_loss_fpops_fn(
        target_fpops=1.1 * true_fpops,
        orbitalized_gm=tr_model,
        input_shapes=(input_shape,),
        device=device,
        unit="kmapps",
    )
    fpops_loss = fpops_loss_fn(tr_model)
    _assert_close(fpops_loss, torch.tensor(0.0, dtype=DTYPE))

    target_fpops_factor = 0.8
    fpops_loss_fn = ptprun.losses.make_loss_fpops_fn(
        target_fpops=target_fpops_factor * true_fpops,
        orbitalized_gm=tr_model,
        input_shapes=(input_shape,),
        device=device,
        unit="kmapps",
    )
    fpops_loss = fpops_loss_fn(tr_model)
    fpops_loss_expected = torch.tensor(1.0 / target_fpops_factor - 1.0, dtype=DTYPE)
    _assert_close(fpops_loss, fpops_loss_expected)


TEST_COMPARE_LOSSES_AGAINST_NODE_API = [
    ["alexnet", sample_models.alexnet(num_classes=10), (1, 3, 224, 224), 1979204500],
    # Node calculates global mean pool as 0 fpops, hence + 25088 to fix this
    [
        "resnet18",
        sample_models.resnet18(num_classes=10),
        (1, 3, 224, 224),
        3637143040 + 25088,
    ],
]


@pytest.mark.parametrize(
    "model_name, model, input_shape, flops_expected",
    TEST_COMPARE_LOSSES_AGAINST_NODE_API,
)
def test_make_calc_fpops_fn_against_node_api(
    model_name, model, input_shape, flops_expected
):
    tr_model = torch.fx.symbolic_trace(model)
    tr_model.meta["orbits"] = []
    device = torch.device("cpu")
    flops_expected = torch.tensor(flops_expected, dtype=DTYPE)

    calc_fpops_fn = ptprun.losses.make_calc_fpops_fn(
        orbitalized_gm=tr_model,
        input_shapes=(input_shape,),
        device=device,
        unit="flops",
    )

    flops = calc_fpops_fn(tr_model)
    _assert_close(flops, flops_expected)
