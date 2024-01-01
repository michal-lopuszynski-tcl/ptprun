import operator

import torch

from ptprun.orbit import Orbit


def get_dag4_data():
    dag = torch.fx.symbolic_trace(lambda x: x)
    input = next(iter(dag.graph.nodes))
    output = next(iter(reversed(dag.graph.nodes)))

    op = torch.nn.Conv2d(
        in_channels=3, out_channels=8, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv1", op)

    op = torch.nn.Conv2d(
        in_channels=3, out_channels=8, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv2", op)

    op = torch.nn.Conv2d(
        in_channels=3, out_channels=16, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv3", op)

    op = torch.nn.Conv2d(
        in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv4", op)

    with dag.graph.inserting_before(output):
        # conv1 = nd.ops.Conv2D.from_conv_op_params(in_channels=3, out_channels=8, filter_size=3, name='conv1')
        # conv2 = nd.ops.Conv2D.from_conv_op_params(in_channels=3, out_channels=8, filter_size=3, name='conv2')
        # conv3 = nd.ops.Conv2D.from_conv_op_params(in_channels=3, out_channels=16, filter_size=3, name='conv3')
        # conv1, conv2, conv3 = cell.register_nodes_parallelly(nodes=[conv1, conv2, conv3])
        conv1 = dag.graph.call_module("conv1", args=(input,))
        conv1.name = "conv1"
        conv2 = dag.graph.call_module("conv2", args=(input,))
        conv2.name = "conv2"
        conv3 = dag.graph.call_module("conv3", args=(input,))
        conv3.name = "conv3"

        # concat1 = nd.ops.Concat(name='concat1', axis=-1)
        # concat1 = cell.register_node(node=concat1, predecessors=[conv1, conv2])

        concat1 = dag.graph.call_function(
            torch.cat, args=([conv1, conv2],), kwargs={"dim": 1}
        )
        concat1.name = "concat1"

        # sum1 = nd.ops.Sum(name='sum1')
        # sum1 = cell.register_node(node=sum1, predecessors=[concat1, conv3])
        sum1 = dag.graph.call_function(operator.add, args=(concat1, conv3))
        sum1.name = "sum1"

        # conv4 = nd.ops.Conv2D.from_conv_op_params(in_channels=16, out_channels=32, filter_size=3, name='conv4')
        # conv4 = cell.register_node(node=conv4, predecessors=[sum1])
        conv4 = dag.graph.call_module("conv4", args=(sum1,))
        # output = nd.ops.TensorMerger(name='output')
        # output = cell.register_node(node=output, predecessors=[conv4])
        # cell.mark_current_top_node_as_output()
        output.args = (conv4,)

    dag.recompile()

    # EXTENDED ORBITS
    extended_orbit1 = Orbit(color=0)
    extended_orbit2 = Orbit(color=1)

    extended_orbit1.sources = [conv1, conv2, conv3]
    extended_orbit1.sinks = [conv4]
    extended_orbit1.icns_in_scope = set(
        extended_orbit1.sources + [concat1, sum1] + extended_orbit1.sinks
    )

    extended_orbit2.sources = [conv4]
    extended_orbit2.sinks = []
    extended_orbit2.icns_in_scope = set(
        extended_orbit2.sources + [output] + extended_orbit2.sinks
    )

    # Old version of extended orbits
    # extended_orbits = [extended_orbit1, extended_orbit2]
    extended_orbits = [extended_orbit2]

    # FINAL ORBITS
    final_orbits = []
    return dag, extended_orbits, final_orbits
