import logging

import pthelpers as pth
import pytest
import torch

import ptprun

import sample_dags  # isort: skip

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("bias", [True, False])
def test_make_pruned_conv2d(bias):
    generator = torch.Generator().manual_seed(12345)

    m = torch.nn.Conv2d(
        in_channels=7, out_channels=11, kernel_size=(3, 3), bias=bias, padding="same"
    )

    input_mask = torch.tensor(
        [True, False, True, False, True, True, False],
        dtype=torch.bool,
        requires_grad=False,
    )
    output_mask = torch.tensor(
        [True, True, True, False, True, False, False, False, False, False, True],
        dtype=torch.bool,
        requires_grad=False,
    )
    m_pruned = ptprun.distillation._make_distilled_conv2d_plain(
        m, output_mask=output_mask, predecessor_masks=[input_mask]
    )

    for _ in range(10):
        x_inp = torch.rand(1, 7, 32, 32, generator=generator)

        x_inp_masked = torch.zeros_like(x_inp)
        for b in range(x_inp.shape[0]):
            for c in range(x_inp.shape[1]):
                if input_mask[c]:
                    x_inp_masked[b, c, :, :] = x_inp[b, c, :, :]

        x_inp_pruned = x_inp[:, input_mask, :, :]

        y1 = m(x_inp_masked)[:, output_mask, :, :].detach()
        y2 = m_pruned(x_inp_pruned).detach()
        res = torch.max(torch.abs(y1 - y2)).item()
        assert res < 1.0e-6


def _count_params(module):
    return sum(dict((p.data_ptr(), p.numel()) for p in module.parameters()).values())


def _get_random_orbit_mask_fn(graph_module, orbit, generator):
    module_dict = dict(graph_module.named_modules())
    mod = module_dict[orbit.sources[0].target]
    num_channels = mod.out_channels
    mask = torch.rand(num_channels, generator=generator) > 0.5
    if sum(mask) <= 0:
        i = torch.randint(size=(1,), low=0, high=num_channels, dtype=torch.int32).item()
        mask = torch.zeros(num_channels, dtype=torch.bool)
        mask[i] = True
    assert sum(mask) > 0
    return mask


def _add_batchnorms_to_graph(gm):
    module_dict = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if pth.fxfilters.is_conv2d_plain(node, module_dict):
            print(f"Inserting batch-norm before {node.target}")
            conv2d = module_dict[node.target]
            bn2d_module = torch.nn.BatchNorm2d(num_features=conv2d.in_channels)
            bn2d_name = f"{node.target}_bn"
            gm.add_module(bn2d_name, bn2d_module)

            with gm.graph.inserting_before(node):
                n_bn = gm.graph.call_module(module_name=bn2d_name, args=node.args)
            node.update_arg(0, n_bn)
    gm.recompile()


def _add_batchnorms_to_orbits(gm, orbits):
    module_dict = dict(gm.named_modules())

    for o in orbits:
        for node in o.sinks:
            if pth.fxfilters.is_conv2d_plain(node, module_dict):
                bn_name = node.target + "_bn"
                bn_node = pth.get_fxnode(gm, bn_name)
                o.add_to_scope(bn_node)


def _make_data_factory_with_bn_fn(data_factory_fn):
    def _data_factory_with_bn_fn():
        dag, eo, fo = data_factory_fn()
        _add_batchnorms_to_graph(dag)
        _add_batchnorms_to_orbits(dag, eo)
        _add_batchnorms_to_orbits(dag, fo)
        return dag, eo, fo

    return _data_factory_with_bn_fn


def _make_orbits_data():
    res = []
    for k, data_factory_fn in sample_dags.DAG_DATA_FACTORIES.items():
        _, _, fo = data_factory_fn()
        if len(fo) > 0:
            res.append([k, data_factory_fn])
            res.append([k + "_bn", _make_data_factory_with_bn_fn(data_factory_fn)])
    return res


@pytest.mark.parametrize("name, dag_data_factory_fn", _make_orbits_data())
def test_distilling(name, dag_data_factory_fn):
    generator = torch.Generator().manual_seed(12345)
    x = torch.rand((1, 3, 224, 224), generator=generator)

    def get_orbit_mask_fn(graph_module, orbit):
        return _get_random_orbit_mask_fn(graph_module, orbit, generator)

    for i in range(30):
        graph_module, _, final_orbits = dag_data_factory_fn()
        logger.info(f"O raw: {final_orbits}")
        # final_orbits = ptprun.extraction.remove_orbits_inp_out(final_orbits)
        logger.info(f"O cleaned: {final_orbits}")
        num_params_i = _count_params(graph_module)
        y_i = graph_module(x)
        logger.info(f"{name} {y_i[0].shape}")
        node_to_mask = ptprun.distillation.get_orbits_distilling_mask(
            graph_module, final_orbits, get_orbit_mask_fn
        )
        ptprun.distillation.distill_module_in_place(graph_module, node_to_mask)
        y_f = graph_module(x)
        num_params_f = _count_params(graph_module)
        logger.info(f"{name}, {i} pruning params {num_params_i} -> {num_params_f}")
        assert num_params_f < num_params_i
        if isinstance(y_i, tuple):
            assert len(y_i) == len(y_f)
            for y_ii, y_ff in zip(y_i, y_f):
                assert y_ii.shape == y_ff.shape
        else:
            assert y_i.shape == y_f.shape
