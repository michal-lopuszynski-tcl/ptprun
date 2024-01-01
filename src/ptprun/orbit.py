import collections.abc

import torch


class Orbit:
    def __init__(self, color: int) -> None:
        self.color = color

        self.icns_in_scope: set[torch.fx.Node] = set()
        self.sources: list[torch.fx.Node] = []
        self.sinks: list[torch.fx.Node] = []
        self.end_path: list[tuple[torch.fx.Node, torch.fx.Node]] = []

    @property
    def non_border(self) -> set[torch.fx.Node]:
        return self.icns_in_scope - (set(self.sources) | set(self.sinks))

    def __repr__(self) -> str:
        return (
            f"Orbit color={self.color}, "
            f"sources={self.sources}, "
            f"sinks={self.sinks}, "
            f"non_border={self.non_border}, "
            f"end_path={self.end_path}]"
        )

    def __iter__(self) -> collections.abc.Iterable[torch.fx.Node]:
        yield from self.icns_in_scope

    def __len__(self) -> int:
        return len(self.icns_in_scope)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Orbit):
            sources_equal = set(self.sources) == set(other.sources)
            sinks_equal = set(self.sinks) == set(other.sinks)
            non_border_equal = self.non_border == other.non_border

            return sources_equal and sinks_equal and non_border_equal
        else:
            return False

    def add_to_scope(self, icn: torch.fx.Node) -> None:
        self.icns_in_scope.add(icn)

    def mark_as_source(self, icn: torch.fx.Node) -> None:
        if icn not in self.sources:
            self.sources += [icn]

    def mark_as_sink(self, icn: torch.fx.Node) -> None:
        if icn not in self.sinks:
            self.sinks += [icn]

    def mark_end_path_node_and_sink(
        self, end_icn: torch.fx.Node, sink_icn: torch.fx.Node
    ) -> None:
        self.end_path += [(end_icn, sink_icn)]


def is_orbit_with_node_name(orbit: Orbit, names: str | tuple[str]) -> bool:
    if isinstance(names, str):
        names = (names,)
    for n in orbit.icns_in_scope:
        if n.name in names:
            return True
    return False
