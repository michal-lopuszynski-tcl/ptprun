import torch

from ptprun.orbit import Orbit

from .commons import get_node


class Dag2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding="same"
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding="same"
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding="same"
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding="same"
        )
        self.conv5 = torch.nn.Conv2d(
            in_channels=56, out_channels=64, kernel_size=3, padding="same"
        )

    def forward(self, x):
        # conv1 = nd.ops.Conv2D.from_conv_op_params(in_channels=3, out_channels=8, filter_size=3, name='conv1')
        # conv1 = cell.register_node(node=conv1)
        x_conv1 = self.conv1(x)

        # act_conv1 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv1')
        # act_conv1 = cell.register_node(node=act_conv1, predecessors=[conv1])
        x_act_conv1 = torch.nn.functional.relu(x_conv1)

        # conv2 = nd.ops.Conv2D.from_conv_op_params(in_channels=8, out_channels=16, filter_size=3, name='conv2')
        # conv3 = nd.ops.Conv2D.from_conv_op_params(in_channels=8, out_channels=16, filter_size=3, name='conv3')
        # conv2, conv3 = cell.register_nodes_parallelly(nodes=[conv2, conv3], predecessors=[act_conv1])
        x_conv2 = self.conv2(x_act_conv1)
        x_conv3 = self.conv3(x_act_conv1)

        # act_conv2 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv2')
        # act_conv2 = cell.register_node(node=act_conv2, predecessors=[conv2])
        x_act_conv2 = torch.nn.functional.relu(x_conv2)

        # act_conv3 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv3')
        # act_conv3 = cell.register_node(node=act_conv3, predecessors=[conv3])
        x_act_conv3 = torch.nn.functional.relu(x_conv3)

        # conv4 = nd.ops.Conv2D.from_conv_op_params(in_channels=16, out_channels=32, filter_size=3, name='conv4')
        # conv4 = cell.register_node(node=conv4, predecessors=[act_conv3])
        x_conv4 = self.conv4(x_act_conv3)

        # act_conv4 = nd.ops.Activation.from_activation_name(activation_name='relu', name='act_conv4')
        # act_conv4 = cell.register_node(node=act_conv4, predecessors=[conv4])
        x_act_conv4 = torch.nn.functional.relu(x_conv4)

        # concat1 = nd.ops.Concat(name='concat1', axis=-1)
        # concat1 = cell.register_node(node=concat1, predecessors=[act_conv2, act_conv4, act_conv1])
        x_concat1 = torch.cat([x_act_conv2, x_act_conv4, x_act_conv1], dim=1)

        # conv5 = nd.ops.Conv2D.from_conv_op_params(in_channels=56, out_channels=64, filter_size=3, name='conv5')
        # conv5 = cell.register_node(node=conv5, predecessors=[concat1])
        x_conv5 = self.conv5(x_concat1)
        return x_conv5


def get_dag2_data():
    dag_raw = Dag2()
    dag = torch.fx.symbolic_trace(dag_raw)

    conv1 = get_node(dag, "conv1")
    conv2 = get_node(dag, "conv2")
    conv3 = get_node(dag, "conv3")
    conv4 = get_node(dag, "conv4")
    conv5 = get_node(dag, "conv5")
    act_conv1 = get_node(dag, "relu")
    act_conv2 = get_node(dag, "relu_1")
    act_conv3 = get_node(dag, "relu_2")
    act_conv4 = get_node(dag, "relu_3")
    concat1 = get_node(dag, "cat")
    output = get_node(dag, "output")

    # EXTENDED ORBITS
    extended_orbit1 = Orbit(color=1)

    extended_orbit1.sources = [conv1, conv2, conv4]
    extended_orbit1.sinks = [conv2, conv3, conv5]
    extended_orbit1.icns_in_scope = set(
        extended_orbit1.sources
        + [act_conv1, act_conv2, act_conv4, concat1]
        + extended_orbit1.sinks
    )

    extended_orbit2 = Orbit(color=2)
    extended_orbit2.sources = [conv3]
    extended_orbit2.sinks = [conv4]
    extended_orbit2.icns_in_scope = set(
        extended_orbit2.sources + [act_conv3] + extended_orbit2.sinks
    )

    extended_orbit3 = Orbit(color=3)
    extended_orbit3.sources = [conv5]
    extended_orbit3.sinks = []
    extended_orbit3.icns_in_scope = set(
        extended_orbit3.sources + [output] + extended_orbit3.sinks
    )

    extended_orbits = [extended_orbit1, extended_orbit2, extended_orbit3]

    # FINAL ORBITS
    final_orbit1 = Orbit(color=1)
    final_orbit1.sources = [conv1]
    final_orbit1.sinks = [conv2, conv3]
    final_orbit1.icns_in_scope = set(
        final_orbit1.sources + [act_conv1] + final_orbit1.sinks
    )

    final_orbit2 = Orbit(color=2)
    final_orbit2.sources = [conv3]
    final_orbit2.sinks = [conv4]
    final_orbit2.icns_in_scope = set(
        final_orbit2.sources + [act_conv3] + final_orbit2.sinks
    )

    final_orbit3 = Orbit(color=3)
    final_orbit3.sources = [conv2]
    final_orbit3.sinks = []
    final_orbit3.icns_in_scope = set(
        final_orbit3.sources + [act_conv2] + final_orbit3.sinks
    )

    final_orbit4 = Orbit(color=4)
    final_orbit4.sources = [conv4]
    final_orbit4.sinks = []
    final_orbit4.icns_in_scope = set(
        final_orbit4.sources + [act_conv4] + final_orbit4.sinks
    )

    final_orbits = [final_orbit1, final_orbit2, final_orbit3, final_orbit4]

    return dag, extended_orbits, final_orbits
