import logging

import pytest
import torch

import ptprun.orbitalization

import sample_dags  # isort:skip

logger = logging.getLogger(__name__)


TEST_ORBITS_DATA = [[k, v] for k, v in sample_dags.DAG_DATA_FACTORIES.items()]


@pytest.mark.parametrize("graph_name, data_factory_fn", TEST_ORBITS_DATA)
def test_orbitalize_in_place(graph_name, data_factory_fn):
    graph_module, _, final_orbits = data_factory_fn()
    generator = torch.Generator().manual_seed(12345)
    device = torch.device("cpu")
    x = torch.rand((1, 3, 224, 224), generator=generator)

    y_i = graph_module(x)
    ptprun.orbitalization.orbitalize_in_place(graph_module, final_orbits, device)
    y_f = graph_module(x)

    if isinstance(y_i, tuple):
        assert len(y_i) == len(y_f)
        for i, (y_ii, y_ff) in enumerate(zip(y_i, y_f)):
            logger.info(f"{graph_name},  out_shape[{i}]: {y_ii.shape} -> {y_ff.shape}")
            assert y_ii.shape == y_ff.shape
    else:
        logger.info(f"{graph_name}, out_shape: {y_i.shape} -> {y_f.shape}")
        assert y_i.shape == y_f.shape


@pytest.mark.parametrize("graph_name, data_factory_fn", TEST_ORBITS_DATA)
def test_deorbitalize_in_place(graph_name, data_factory_fn):
    graph_module, _, final_orbits = data_factory_fn()

    generator = torch.Generator().manual_seed(12345)
    device = torch.device("cpu")
    x = torch.rand((1, 3, 224, 224), generator=generator)

    y_i = graph_module(x)
    node_names_i = [node.name for node in graph_module.graph.nodes]
    ptprun.orbitalization.orbitalize_in_place(graph_module, final_orbits, device)
    ptprun.orbitalization.deorbitalize_in_place(graph_module)
    node_names_f = [node.name for node in graph_module.graph.nodes]
    y_f = graph_module(x)

    assert len(node_names_f) == len(node_names_i)
    assert node_names_i == node_names_f
    if isinstance(y_i, tuple):
        assert len(y_i) == len(y_f)
        for i, (y_ii, y_ff) in enumerate(zip(y_i, y_f)):
            torch.testing.assert_close(y_ii, y_ff)
    else:
        logger.info(f"{graph_name}, out_shape: {y_i.shape} -> {y_f.shape}")
        torch.testing.assert_close(y_i, y_f)
