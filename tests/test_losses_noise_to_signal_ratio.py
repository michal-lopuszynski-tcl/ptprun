import numpy as np
import pytest
import torch

import ptprun


def _gen_random_x_and_y(*, seed, size, transpose=None):
    rng = np.random.default_rng(seed=seed)
    while True:
        x = rng.normal(size=size).astype(np.float32)
        y = rng.normal(size=size).astype(np.float32)
        if transpose is None:
            yield x, y
        else:
            yield np.transpose(x, axes=transpose), np.transpose(y, axes=transpose)


def _make_data():
    result_expected_list = [1.9850197, 1.9766895, 1.9759563, 2.0157917, 2.0182467]
    gen_x_y_res = zip(
        _gen_random_x_and_y(seed=12345, size=(5, 32, 32, 3), transpose=[0, 3, 1, 2]),
        result_expected_list,
    )
    return [
        (torch.tensor(x), torch.tensor(y), torch.tensor(result_expected))
        for (x, y), result_expected in gen_x_y_res
    ]


@pytest.mark.parametrize("x, y, result_expected", _make_data())
def test_calc_noise_to_signal_ratio(x, y, result_expected):
    ## Expected results generated with the following code:
    ## node_api_research version 0dc40899ca1acfaf0d35e5149048484611488f03
    #
    # import node_api_research as ndr
    #
    # for  x, y in _gen_random_x_and_y(seed=12345, n=5, size=(5,32,32,3)):
    #     loss = ndr.utils.per_channel_noise_to_signal_ratio(tf.constant(x), tf.constant(y))
    # print(loss.numpy())

    result = ptprun.losses.calc_noise_to_signal_ratio(x, y)
    torch.testing.assert_close(result, result_expected)
