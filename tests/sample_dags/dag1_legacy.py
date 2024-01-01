import torch.fx
import torch.nn
import torch.nn.functional

from ptprun.orbit import Orbit

from .commons import get_node


class Dag1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding="same"
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding="same"
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding="same"
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding="same"
        )
        self.conv5 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding="same"
        )
        self.conv6 = torch.nn.Conv2d(
            in_channels=24, out_channels=32, kernel_size=3, padding="same"
        )
        self.conv7 = torch.nn.Conv2d(
            in_channels=8, out_channels=32, kernel_size=3, padding="same"
        )
        self.conv8 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding="same"
        )

    def forward(self, x):
        # conv1 = nd.ops.Conv2D.from_conv_op_params(in_channels=3, out_channels=8, filter_size=3, name='conv1')
        # conv2 = nd.ops.Conv2D.from_conv_op_params(in_channels=3, out_channels=8, filter_size=3, name='conv2')
        # conv3 = nd.ops.Conv2D.from_conv_op_params(in_channels=3, out_channels=8, filter_size=3, name='conv3')
        # conv1, conv2, conv3 = cell.register_nodes_parallelly(nodes=[conv1, conv2, conv3])

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x)
        x_conv3 = self.conv3(x)

        # act_conv1 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv1')
        # act_conv1 = cell.register_node(node=act_conv1, predecessors=[conv1])

        x_act_conv1 = torch.nn.functional.relu(x_conv1)

        # conv4 = nd.ops.Conv2D.from_conv_op_params(in_channels=8, out_channels=16, filter_size=3, name='conv4')
        # conv4 = cell.register_node(node=conv4, predecessors=[act_conv1])
        # act_conv4 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv4')
        # act_conv4 = cell.register_node(node=act_conv4, predecessors=[conv4])

        x_conv4 = self.conv4(x_act_conv1)
        x_act_conv4 = torch.nn.functional.relu(x_conv4)

        # conv5 = nd.ops.Conv2D.from_conv_op_params(in_channels=16, out_channels=32, filter_size=3, name='conv5')
        # conv5 = cell.register_node(node=conv5, predecessors=[act_conv4])
        x_conv5 = self.conv5(x_act_conv4)

        # act_conv2 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv2')
        # act_conv2 = cell.register_node(node=act_conv2, predecessors=[conv2])

        x_act_conv2 = torch.nn.functional.relu(x_conv2)

        # act1_conv3 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act1_conv3')
        # act2_conv3 = nd.ops.Activation.from_activation_name(activation_name='sigmoid', name='act2_conv3')
        # act1_conv3, act2_conv3 = cell.register_nodes_parallelly(nodes=[act1_conv3, act2_conv3], predecessors=[conv3])
        x_act1_conv3 = torch.nn.functional.relu(x_conv3)
        x_act2_conv3 = torch.sigmoid(x_conv3)

        # conv7 = nd.ops.Conv2D.from_conv_op_params(in_channels=8, out_channels=32, filter_size=3, name='conv7')
        # conv7 = cell.register_node(node=conv7, predecessors=[act2_conv3])
        x_conv7 = self.conv7(x_act2_conv3)

        # concat1 = nd.ops.Concat(name='concat1', axis=-1)
        # concat1 = cell.register_node(node=concat1, predecessors=[act_conv2, act1_conv3, act2_conv3])
        x_concat1 = torch.cat([x_act_conv2, x_act1_conv3, x_act2_conv3], dim=1)

        # conv6 = nd.ops.Conv2D.from_conv_op_params(in_channels=24, out_channels=32, filter_size=3, name='conv6')
        # conv6 = cell.register_node(node=conv6, predecessors=[concat1])
        x_conv6 = self.conv6(x_concat1)

        # concat2 = nd.ops.Concat(name='concat2', axis=-1)
        # concat2 = cell.register_node(node=concat2, predecessors=[act_conv1, act_conv2])
        x_concat2 = torch.cat([x_act_conv1, x_act_conv2], dim=1)

        # conv8 = nd.ops.Conv2D.from_conv_op_params(in_channels=16, out_channels=32, filter_size=3, name='conv8')
        # conv8 = cell.register_node(node=conv8, predecessors=[concat2])
        x_conv8 = self.conv8(x_concat2)

        # sum1 = nd.ops.Sum(name='sum1')
        # sum1 = cell.register_node(node=sum1, predecessors=[conv5, conv8])
        x_sum1 = x_conv5 + x_conv8

        # output = nd.ops.TensorMerger(name='output')
        # output = cell.register_node(node=output, predecessors=[conv5, sum1, conv6, conv7])
        # cell.mark_current_top_node_as_output()
        return x_conv5, x_sum1, x_conv6, x_conv7


def get_dag1_data():
    dag1_raw = Dag1()
    dag1 = torch.fx.symbolic_trace(dag1_raw)

    # EXTENDED ORBITS
    extended_orbit1 = Orbit(color=1)
    extended_orbit2 = Orbit(color=2)
    extended_orbit3 = Orbit(color=3)

    conv1 = get_node(dag1, "conv1")
    conv2 = get_node(dag1, "conv2")
    conv3 = get_node(dag1, "conv3")
    conv4 = get_node(dag1, "conv4")
    conv5 = get_node(dag1, "conv5")
    conv6 = get_node(dag1, "conv6")
    conv7 = get_node(dag1, "conv7")
    conv8 = get_node(dag1, "conv8")
    act_conv1 = get_node(dag1, "relu")
    act_conv2 = get_node(dag1, "relu_2")
    act1_conv3 = get_node(dag1, "relu_3")
    act2_conv3 = get_node(dag1, "sigmoid")
    act_conv4 = get_node(dag1, "relu_1")
    concat1 = get_node(dag1, "cat")
    concat2 = get_node(dag1, "cat_1")
    sum1 = get_node(dag1, "add")
    output = get_node(dag1, "output")

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
    final_orbit1 = Orbit(color=1)
    final_orbit2 = Orbit(color=2)
    final_orbit3 = Orbit(color=3)
    final_orbit4 = Orbit(color=4)

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

    return dag1, extended_orbits, final_orbits
