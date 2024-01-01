import operator

import torch

from ptprun.orbit import Orbit


def get_dag1_data():
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
        in_channels=3, out_channels=8, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv3", op)

    op = torch.nn.Conv2d(
        in_channels=8, out_channels=16, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv4", op)

    op = torch.nn.Conv2d(
        in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv5", op)

    op = torch.nn.Conv2d(
        in_channels=24, out_channels=32, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv6", op)

    op = torch.nn.Conv2d(
        in_channels=8, out_channels=32, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv7", op)

    op = torch.nn.Conv2d(
        in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv8", op)

    with dag.graph.inserting_before(output):
        # conv1 = nd.ops.Conv2D.from_conv_op_params(in_channels=3, out_channels=8, filter_size=3, name='conv1')
        # conv2 = nd.ops.Conv2D.from_conv_op_params(in_channels=3, out_channels=8, filter_size=3, name='conv2')
        # conv3 = nd.ops.Conv2D.from_conv_op_params(in_channels=3, out_channels=8, filter_size=3, name='conv3')
        # conv1, conv2, conv3 = cell.register_nodes_parallelly(nodes=[conv1, conv2, conv3])
        conv1 = dag.graph.call_module("conv1", args=(input,))
        conv1.name = "conv1"
        conv2 = dag.graph.call_module("conv2", args=(input,))
        conv2.name = "conv2"
        conv3 = dag.graph.call_module("conv3", args=(input,))
        conv3.name = "conv3"

        # act_conv1 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv1')
        # act_conv1 = cell.register_node(node=act_conv1, predecessors=[conv1])
        act_conv1 = dag.graph.call_function(torch.nn.functional.relu, args=(conv1,))
        act_conv1.name = "act_conv1"

        # conv4 = nd.ops.Conv2D.from_conv_op_params(in_channels=8, out_channels=16, filter_size=3, name='conv4')
        # conv4 = cell.register_node(node=conv4, predecessors=[act_conv1])
        conv4 = dag.graph.call_module("conv4", args=(act_conv1,))
        conv4.name = "conv4"

        # act_conv4 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv4')
        # act_conv4 = cell.register_node(node=act_conv4, predecessors=[conv4])
        act_conv4 = dag.graph.call_function(torch.nn.functional.relu, args=(conv4,))
        act_conv4.name = "act_conv4"

        # conv5 = nd.ops.Conv2D.from_conv_op_params(in_channels=16, out_channels=32, filter_size=3, name='conv5')
        # conv5 = cell.register_node(node=conv5, predecessors=[act_conv4])
        conv5 = dag.graph.call_module("conv5", args=(act_conv4,))
        conv5.name = "conv5"

        # act_conv2 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv2')
        # act_conv2 = cell.register_node(node=act_conv2, predecessors=[conv2])
        act_conv2 = dag.graph.call_function(torch.nn.functional.relu, args=(conv2,))
        act_conv2.name = "act_conv2"

        # act1_conv3 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act1_conv3')
        # act2_conv3 = nd.ops.Activation.from_activation_name(activation_name='sigmoid', name='act2_conv3')
        # act1_conv3, act2_conv3 = cell.register_nodes_parallelly(nodes=[act1_conv3, act2_conv3], predecessors=[conv3])
        act1_conv3 = dag.graph.call_function(torch.nn.functional.relu, args=(conv3,))
        act1_conv3.name = "act1_conv3"
        act2_conv3 = dag.graph.call_function(torch.sigmoid, args=(conv3,))
        act2_conv3.name = "act2_conv3"

        # conv7 = nd.ops.Conv2D.from_conv_op_params(in_channels=8, out_channels=32, filter_size=3, name='conv7')
        # conv7 = cell.register_node(node=conv7, predecessors=[act2_conv3])
        conv7 = dag.graph.call_module("conv7", args=(act2_conv3,))
        conv7.name = "conv7"

        # concat1 = nd.ops.Concat(name='concat1', axis=-1)
        # concat1 = cell.register_node(node=concat1, predecessors=[act_conv2, act1_conv3, act2_conv3])
        concat1 = dag.graph.call_function(
            torch.cat, args=([act_conv2, act1_conv3, act2_conv3],), kwargs={"dim": 1}
        )
        concat1.name = "concat1"

        # conv6 = nd.ops.Conv2D.from_conv_op_params(in_channels=24, out_channels=32, filter_size=3, name='conv6')
        # conv6 = cell.register_node(node=conv6, predecessors=[concat1])
        conv6 = dag.graph.call_module("conv6", args=(concat1,))
        conv6.name = "conv6"

        # concat2 = nd.ops.Concat(name='concat2', axis=-1)
        # concat2 = cell.register_node(node=concat2, predecessors=[act_conv1, act_conv2])
        concat2 = dag.graph.call_function(
            torch.cat, args=([act_conv1, act_conv2],), kwargs={"dim": 1}
        )
        concat2.name = "concat2"

        # conv8 = nd.ops.Conv2D.from_conv_op_params(in_channels=16, out_channels=32, filter_size=3, name='conv8')
        # conv8 = cell.register_node(node=conv8, predecessors=[concat2])
        conv8 = dag.graph.call_module("conv8", args=(concat2,))
        conv8.name = "conv8"

        # sum1 = nd.ops.Sum(name='sum1')
        # sum1 = cell.register_node(node=sum1, predecessors=[conv5, conv8])
        sum1 = dag.graph.call_function(operator.add, args=(conv5, conv8))
        sum1.name = "sum1"

        # output = nd.ops.TensorMerger(name='output')
        # output = cell.register_node(node=output, predecessors=[conv5, sum1, conv6, conv7])
        # cell.mark_current_top_node_as_output()
        output.args = ((conv5, sum1, conv6, conv7),)

    dag.recompile()

    # EXTENDED ORBITS

    extended_orbit1 = Orbit(color=0)
    extended_orbit2 = Orbit(color=1)
    extended_orbit3 = Orbit(color=2)

    extended_orbit1.sources = [conv1, conv2, conv3]
    extended_orbit1.sinks = [conv4, conv8, conv6, conv7]
    extended_orbit1.icns_in_scope = set(
        extended_orbit1.sources
        + [act_conv1, act_conv2, act1_conv3, act2_conv3, concat2, concat1]
        + extended_orbit1.sinks
    )

    extended_orbit2.sources = [conv4]
    extended_orbit2.sinks = [conv5]
    extended_orbit2.icns_in_scope = set(
        extended_orbit2.sources + [act_conv4] + extended_orbit2.sinks
    )

    extended_orbit3.sources = [conv5, conv8, conv6, conv7]
    extended_orbit3.sinks = []
    extended_orbit3.icns_in_scope = set(
        extended_orbit3.sources + [sum1, output] + extended_orbit3.sinks
    )

    extended_orbits = [extended_orbit1, extended_orbit2, extended_orbit3]

    # FINAL ORBITS

    final_orbit1 = Orbit(color=0)
    final_orbit2 = Orbit(color=1)
    final_orbit3 = Orbit(color=2)
    final_orbit4 = Orbit(color=3)

    final_orbit1.sources = [conv1]
    final_orbit1.sinks = [conv4]
    final_orbit1.icns_in_scope = set(
        final_orbit1.sources + [act_conv1] + final_orbit1.sinks
    )

    final_orbit2.sources = [conv2]
    final_orbit2.sinks = []
    final_orbit2.icns_in_scope = set(
        final_orbit2.sources + [act_conv2] + final_orbit2.sinks
    )

    final_orbit3.sources = [conv3]
    final_orbit3.sinks = [conv7]
    final_orbit3.icns_in_scope = set(
        final_orbit3.sources + [act2_conv3, act1_conv3] + final_orbit3.sinks
    )

    final_orbit4.sources = [conv4]
    final_orbit4.sinks = [conv5]
    final_orbit4.icns_in_scope = set(
        final_orbit4.sources + [act_conv4] + final_orbit4.sinks
    )

    final_orbits = [final_orbit1, final_orbit2, final_orbit3, final_orbit4]
    return dag, extended_orbits, final_orbits
