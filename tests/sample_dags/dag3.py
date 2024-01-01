import operator

import torch

from ptprun.orbit import Orbit


def get_dag3_data():
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
        in_channels=8, out_channels=8, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv3", op)

    op = torch.nn.Conv2d(
        in_channels=8, out_channels=32, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv4", op)

    op = torch.nn.Conv2d(
        in_channels=32, out_channels=32, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv5", op)

    op = torch.nn.Conv2d(
        in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv6", op)

    op = torch.nn.Conv2d(
        in_channels=72, out_channels=128, kernel_size=(3, 3), padding="same"
    )
    dag.add_module("conv7", op)

    with dag.graph.inserting_before(output):
        # conv1 = nd.ops.Conv2D.from_conv_op_params(in_channels=3, out_channels=8, filter_size=3, name='conv1')
        # conv2 = nd.ops.Conv2D.from_conv_op_params(in_channels=3, out_channels=8, filter_size=3, name='conv2')
        # conv1, conv2 = cell.register_nodes_parallelly(nodes=[conv1, conv2])

        conv1 = dag.graph.call_module("conv1", args=(input,))
        conv2 = dag.graph.call_module("conv2", args=(input,))

        # act_conv1 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv1')
        # act_conv1 = cell.register_node(node=act_conv1, predecessors=[conv1])
        act_conv1 = dag.graph.call_function(torch.nn.functional.relu, args=(conv1,))
        act_conv1.name = "act_conv1"

        # act_conv2 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv2')
        # act_conv2 = cell.register_node(node=act_conv2, predecessors=[conv2])
        act_conv2 = dag.graph.call_function(torch.nn.functional.relu, args=(conv2,))
        act_conv2.name = "act_conv2"

        # conv3 = nd.ops.Conv2D.from_conv_op_params(in_channels=8, out_channels=8, filter_size=3, name='conv3')
        # conv3 = cell.register_node(node=conv3, predecessors=[act_conv2])
        conv3 = dag.graph.call_module("conv3", args=(act_conv2,))

        # act_conv3 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv3')
        # act_conv3 = cell.register_node(node=act_conv3, predecessors=[conv3])

        act_conv3 = dag.graph.call_function(torch.nn.functional.relu, args=(conv3,))
        act_conv3.name = "act_conv3"

        # sum1 = nd.ops.Sum(name='sum1')
        # sum1 = cell.register_node(node=sum1, predecessors=[act_conv3, act_conv2])
        sum1 = dag.graph.call_function(operator.add, args=(act_conv3, act_conv2))
        sum1.name = "sum1"

        # conv4 = nd.ops.Conv2D.from_conv_op_params(in_channels=8, out_channels=32, filter_size=3, name='conv4')
        # conv4 = cell.register_node(node=conv4, predecessors=[sum1])
        conv4 = dag.graph.call_module("conv4", args=(sum1,))
        conv4.name = "conv4"

        # act_conv4 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv4')
        # act_conv4 = cell.register_node(node=act_conv4, predecessors=[conv4])

        act_conv4 = dag.graph.call_function(torch.nn.functional.relu, args=(conv4,))
        act_conv4.name = "act_conv4"

        # conv5 = nd.ops.Conv2D.from_conv_op_params(in_channels=32, out_channels=32, filter_size=3, name='conv5')
        # conv5 = cell.register_node(node=conv5, predecessors=[act_conv4])
        conv5 = dag.graph.call_module("conv5", args=(act_conv4,))
        conv5.name = "conv5"

        # act_conv5 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv5')
        # act_conv5 = cell.register_node(node=act_conv5, predecessors=[conv5])
        act_conv5 = dag.graph.call_function(torch.nn.functional.relu, args=(conv5,))
        act_conv5.name = "act_conv5"

        # sum2 = nd.ops.Sum(name='sum2')
        # sum2 = cell.register_node(node=sum2, predecessors=[act_conv5, act_conv4])
        sum2 = dag.graph.call_function(operator.add, args=(act_conv5, act_conv4))
        sum2.name = "sum2"

        # conv6 = nd.ops.Conv2D.from_conv_op_params(in_channels=32, out_channels=64, filter_size=3, name='conv6')
        # conv6 = cell.register_node(node=conv6, predecessors=[sum2])
        conv6 = dag.graph.call_module("conv6", args=(sum2,))
        conv6.name = "conv6"

        # act_conv6 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv6')
        # act_conv6 = cell.register_node(node=act_conv6, predecessors=[conv6])
        act_conv6 = dag.graph.call_function(torch.nn.functional.relu, args=(conv6,))
        act_conv6.name = "act_conv6"

        # concat1 = nd.ops.Concat(name='concat1', axis=-1)
        # concat1 = cell.register_node(node=concat1, predecessors=[act_conv6, act_conv1])
        concat1 = dag.graph.call_function(
            torch.cat, args=([act_conv6, act_conv1],), kwargs={"dim": 1}
        )
        concat1.name = "concat1"

        # conv7 = nd.ops.Conv2D.from_conv_op_params(in_channels=72, out_channels=128, filter_size=3, name='conv7')
        # conv7 = cell.register_node(node=conv7, predecessors=[concat1])
        conv7 = dag.graph.call_module("conv7", args=(concat1,))
        conv7.name = "conv7"

        # output = nd.ops.TensorMerger(name='output')
        # output = cell.register_node(node=output, predecessors=[conv7])
        # cell.mark_current_top_node_as_output()
        output.args = (conv7,)

    dag.recompile()

    # EXTENDED ORBITS

    extended_orbit1 = Orbit(color=0)
    extended_orbit2 = Orbit(color=1)
    extended_orbit3 = Orbit(color=2)
    extended_orbit4 = Orbit(color=3)

    extended_orbit1.sources = [conv1, conv6]
    extended_orbit1.sinks = [conv7]
    extended_orbit1.icns_in_scope = set(
        extended_orbit1.sources
        + [act_conv1, concat1, act_conv6]
        + extended_orbit1.sinks
    )

    extended_orbit2.sources = [conv2, conv3]
    extended_orbit2.sinks = [conv3, conv4]
    extended_orbit2.icns_in_scope = set(
        extended_orbit2.sources + [act_conv2, act_conv3, sum1] + extended_orbit2.sinks
    )

    extended_orbit3.sources = [conv4, conv5]
    extended_orbit3.sinks = [conv5, conv6]
    extended_orbit3.icns_in_scope = set(
        extended_orbit3.sources + [act_conv4, act_conv5, sum2] + extended_orbit3.sinks
    )

    extended_orbit4.sources = [conv7]
    extended_orbit4.sinks = []
    extended_orbit4.icns_in_scope = set(
        extended_orbit4.sources + [output] + extended_orbit4.sinks
    )

    extended_orbits = [
        extended_orbit1,
        extended_orbit2,
        extended_orbit3,
        extended_orbit4,
    ]

    # FINAL ORBITS
    final_orbit1 = Orbit(color=0)
    final_orbit2 = Orbit(color=1)
    final_orbit3 = Orbit(color=2)
    final_orbit4 = Orbit(color=3)

    final_orbit1.sources = [conv1]
    final_orbit1.sinks = []
    final_orbit1.icns_in_scope = set(
        final_orbit1.sources + [act_conv1] + final_orbit1.sinks
    )

    final_orbit2.sources = [conv6]
    final_orbit2.sinks = []
    final_orbit2.icns_in_scope = set(
        final_orbit2.sources + [act_conv6] + final_orbit2.sinks
    )

    final_orbit3.sources = [conv2, conv3]
    final_orbit3.sinks = [conv3, conv4]
    final_orbit3.icns_in_scope = set(
        final_orbit3.sources + [act_conv2, act_conv3, sum1] + final_orbit3.sinks
    )

    final_orbit4.sources = [conv4, conv5]
    final_orbit4.sinks = [conv5, conv6]
    final_orbit4.icns_in_scope = set(
        final_orbit4.sources + [act_conv4, act_conv5, sum2] + final_orbit4.sinks
    )

    final_orbits = [final_orbit1, final_orbit2, final_orbit3, final_orbit4]

    return dag, extended_orbits, final_orbits
