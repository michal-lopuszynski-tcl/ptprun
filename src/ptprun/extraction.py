import collections.abc
import logging
from typing import Optional

import pthelpers as pth
import torch

from . import coloring, orbit

logger = logging.getLogger(__name__)


class _NodeChecker:
    def __init__(self, module: torch.nn.Module) -> None:
        self.named_modules = dict(module.named_modules())

    def is_source(self, node: torch.fx.Node) -> bool:
        if node.op == "call_module":
            module = self.named_modules[node.target]
            return isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
        return False

    def is_sink(self, node: torch.fx.Node) -> bool:
        if node.op == "call_module":
            module = self.named_modules[node.target]
            return isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
        return False

    def is_truncating(self, node: torch.fx.Node) -> bool:
        return False

    def is_input(self, node: torch.fx.Node) -> bool:
        return node.op == "placeholder"

    def is_output(self, node: torch.fx.Node) -> bool:
        return node.op == "output"


class _TruncatingNodeChecker(_NodeChecker):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__(module)

    def is_truncating(self, node: torch.fx.Node) -> bool:
        if node.op == "call_function" and node.target == torch.cat:
            return True
        return False

    def is_input(self, node: torch.fx.Node) -> bool:
        return node.op == "placeholder"

    def is_output(self, node: torch.fx.Node) -> bool:
        return node.op == "output"


def _find_nodes_in_orbit_scope(
    icn: torch.fx.Node,
    orbit: orbit.Orbit,
    icn_coloring: coloring.Coloring,
    node_checker: _NodeChecker,
) -> orbit.Orbit:
    for predecessor in icn_coloring.available_predecessors(icn, orbit.color):
        if node_checker.is_input(predecessor):
            orbit.add_to_scope(predecessor)
            continue
        if icn_coloring.outgoing_color_is_empty(predecessor):
            orbit.add_to_scope(predecessor)
            icn_coloring.set_outgoing_color(predecessor, orbit.color)
            if node_checker.is_source(predecessor):
                orbit.mark_as_source(predecessor)
            else:
                icn_coloring.set_incoming_color(predecessor, orbit.color)
            _find_nodes_in_orbit_scope(predecessor, orbit, icn_coloring, node_checker)

    for successor in icn_coloring.available_successors(icn, orbit.color):
        if icn_coloring.incoming_color_is_empty(successor):
            if node_checker.is_truncating(successor):
                orbit.mark_end_path_node_and_sink(end_icn=icn, sink_icn=successor)
                continue
            orbit.add_to_scope(successor)
            icn_coloring.set_incoming_color(successor, orbit.color)
            if node_checker.is_sink(successor):
                orbit.mark_as_sink(successor)
                orbit.mark_end_path_node_and_sink(end_icn=icn, sink_icn=successor)
            else:
                icn_coloring.set_outgoing_color(successor, orbit.color)
            _find_nodes_in_orbit_scope(successor, orbit, icn_coloring, node_checker)

    return orbit


def _extract_raw_orbits(
    icns: collections.abc.Iterable[torch.fx.Node],
    node_checker: _NodeChecker,
    sources: Optional[collections.abc.Iterable[torch.fx.Node]] = None,
) -> list[orbit.Orbit]:
    sources = sources if sources else icns
    icn_coloring = coloring.Coloring(icns=icns)

    orbits: list[orbit.Orbit] = []
    for icn in sources:
        if not node_checker.is_input(icn) and node_checker.is_source(icn):
            if not icn_coloring.has_outgoing_color(icn):
                color = icn_coloring.get_next_color()
                icn_coloring.set_outgoing_color(icn, color)
                new_orbit = orbit.Orbit(color=color)
                new_orbit.add_to_scope(icn)
                new_orbit.mark_as_source(icn)
                _find_nodes_in_orbit_scope(icn, new_orbit, icn_coloring, node_checker)
                orbits += [new_orbit]

    return orbits


def _is_join_node(node: torch.fx.Node, module_dict: dict[str, torch.nn.Module]) -> bool:
    return pth.fxfilters.is_mul(node, module_dict) or pth.fxfilters.is_add(
        node, module_dict
    )


def _search_for_join_node(
    node: torch.fx.Node, orbit: orbit.Orbit, module_dict: dict[str, torch.nn.Module]
) -> bool:
    if node in orbit.icns_in_scope and _is_join_node(node, module_dict):
        return True

    if node in orbit.sinks:
        return False

    for successor in node.users:
        if _search_for_join_node(successor, orbit, module_dict):
            return True

    return False


def _contains_join_node_after_concat(
    orbit: orbit.Orbit, module_dict: dict[str, torch.nn.Module]
) -> bool:
    for node in orbit.non_border:
        if pth.fxfilters.is_cat(node, module_dict):
            if _search_for_join_node(node, orbit, module_dict):
                return True
    return False


def _recolor_orbits_in_place(orbits: list[orbit.Orbit]) -> None:
    colors = sorted(o.color for o in orbits)
    mapping = dict(zip(colors, range(len(colors))))

    for o in orbits:
        o.color = mapping[o.color]


def _contains_input_output_node(
    orbit: orbit.Orbit, module_dict: dict[str, torch.nn.Module]
) -> bool:
    for icn in orbit.icns_in_scope:
        if pth.fxfilters.is_input(icn, module_dict) or pth.fxfilters.is_output(
            icn, module_dict
        ):
            return True
    return False


def _is_valid_orbit(o: orbit.Orbit, module_dict: dict[str, torch.nn.Module]) -> bool:
    # # NODE-API-RESEARCH FILTERING

    # single_path_orbits: bool = False,
    # prune_1x1_convs: bool = False,
    # prune_stem: bool = False,

    # extended_orbits_filters = [
    #         filters.JoinOpAfterConcatPresentFilter(),
    #         filters.TensorMergerPresentFilter(),
    #         filters.TokenizationOrDetokenizationPresentFilter(),
    #         filters.SubPixelUpsamplingOrDownsamplingPresetFilter(),
    #         filters.ReshapePresentFilter(),
    # ]

    # if not prune_1x1_convs:
    #     extended_orbits_filters += \
    #           [filters.HasOnly1x1ConvSourcesAndNoDepthwiseInsideFilter()]

    # if single_path_orbits:
    #     extended_orbits_filters += [filters.SinglePathOrbitFilter()]
    if _contains_join_node_after_concat(o, module_dict):
        return False

    for node in o.icns_in_scope:
        if pth.fxfilters.is_flatten(node, module_dict):
            return False

    return True


def _extract_extended_orbits(graph_module: torch.fx.GraphModule) -> list[orbit.Orbit]:
    icns = [n for n in graph_module.graph.nodes]
    module_dict = dict(graph_module.named_modules())

    node_checker = _NodeChecker(graph_module)

    extended_orbits = _extract_raw_orbits(icns=icns, node_checker=node_checker)
    logger.info(f"Found {len(extended_orbits)}")

    extended_orbits = [o for o in extended_orbits if _is_valid_orbit(o, module_dict)]

    logger.info(f"Truncated to {len(extended_orbits)}")
    _recolor_orbits_in_place(extended_orbits)
    return extended_orbits


def extract_orbits(graph_module: torch.fx.GraphModule) -> list[orbit.Orbit]:
    module_dict = dict(graph_module.named_modules())

    extended_orbits = _extract_extended_orbits(graph_module)

    final_orbits = []
    truncating_node_checker = _TruncatingNodeChecker(graph_module)

    for o in extended_orbits:
        final_orbits_from_extended_orbit = _extract_raw_orbits(
            icns=o.icns_in_scope,
            sources=o.sources,
            node_checker=truncating_node_checker,
        )
        final_orbits += final_orbits_from_extended_orbit

    final_orbits = [
        o for o in final_orbits if not _contains_input_output_node(o, module_dict)
    ]
    _recolor_orbits_in_place(final_orbits)
    return final_orbits
