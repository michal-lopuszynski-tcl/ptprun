import collections.abc
import dataclasses
from typing import Optional

import torch


@dataclasses.dataclass
class NodeColor:
    incoming_color: Optional[int] = None
    outgoing_color: Optional[int] = None


def _create_color_generator() -> collections.abc.Generator[int, None, None]:
    i = 0
    while True:
        yield i
        i += 1


class Coloring:
    """
    Class that holds information about incoming and outgoing colors for each node in
    set of inner_cell_nodes.
    INCOMING - incoming edges color
    OUTGOING - outcoming edges color
    """

    color_generator = _create_color_generator()

    def __init__(self, icns: collections.abc.Iterable[torch.fx.Node]) -> None:
        self.colors = self.init_colors(icns)

    def init_colors(
        self, icns: collections.abc.Iterable[torch.fx.Node]
    ) -> dict[torch.fx.Node, NodeColor]:
        return {icn: NodeColor() for icn in icns}

    def has_incoming_color(self, icn: torch.fx.Node) -> bool:
        return True if self.colors[icn].incoming_color is not None else False

    def has_outgoing_color(self, icn: torch.fx.Node) -> bool:
        return True if self.colors[icn].outgoing_color is not None else False

    def in_scope(self, icn: torch.fx.Node) -> bool:
        return icn in self.colors

    def set_incoming_color(self, icn: torch.fx.Node, color: int) -> None:
        self.colors[icn].incoming_color = color

    def set_outgoing_color(self, icn: torch.fx.Node, color: int) -> None:
        self.colors[icn].outgoing_color = color

    def incoming_color_equal_to(self, icn: torch.fx.Node, other_color: int) -> bool:
        return self.colors[icn].incoming_color == other_color

    def outgoing_color_equal_to(self, icn: torch.fx.Node, other_color: int) -> bool:
        return self.colors[icn].outgoing_color == other_color

    def incoming_color_is_empty(self, icn: torch.fx.Node) -> bool:
        return self.in_scope(icn) and not self.has_incoming_color(icn)

    def outgoing_color_is_empty(self, icn: torch.fx.Node) -> bool:
        return self.in_scope(icn) and not self.has_outgoing_color(icn)

    def can_go_backward(self, icn: torch.fx.Node, color: int) -> bool:
        return self.in_scope(icn) and self.incoming_color_equal_to(icn, color)

    def can_go_forward(self, icn: torch.fx.Node, color: int) -> bool:
        return self.in_scope(icn) and self.outgoing_color_equal_to(icn, color)

    def available_predecessors(
        self, icn: torch.fx.Node, color: int
    ) -> list[torch.fx.Node]:
        if self.can_go_backward(icn, color):
            return icn.all_input_nodes

        return []

    def available_successors(
        self, icn: torch.fx.Node, color: int
    ) -> list[torch.fx.Node]:
        if self.can_go_forward(icn, color):
            return list(icn.users.keys())
        return []

    def get_next_color(self) -> int:
        return next(self.color_generator)
