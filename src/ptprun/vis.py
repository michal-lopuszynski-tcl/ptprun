from collections.abc import Callable
from typing import Any, Optional

import graphviz  # type: ignore
import pthelpers as pth
import torch

from .orbit import Orbit

PAL_VEGA_CATEGORY20 = [
    "#1f77b4",  # 00
    "#ff7f0e",  # 02
    "#2ca02c",  # 04
    "#d62728",  # 06
    "#9467bd",  # 08
    "#8c564b",  # 10
    "#e377c2",  # 12
    "#7f7f7f",  # 14
    "#bcbd22",  # 16
    "#17becf",  # 18
    "#aec7e8",  # 01
    "#ffbb78",  # 03
    "#98df8a",  # 05
    "#ff9896",  # 07
    "#c5b0d5",  # 09
    "#c49c94",  # 11
    "#f7b6d2",  # 13
    "#c7c7c7",  # 15
    "#dbdb8d",  # 17
    "#9edae5",  # 19
]


def _get_orbit_color(
    color: int, colors: list[int], palette: list[str], blank_color: str
) -> str:
    if color < 0:
        return blank_color
    else:
        i = colors.index(color)
        return palette[i % len(palette)]


def _make_node_to_orbits_fn(
    orbits: list[Orbit], palette: list[str], blank_color: str
) -> Callable[[torch.fx.Node], tuple[int, int]]:
    node_to_inp_orbit: dict[torch.fx.Node, int] = {}
    node_to_out_orbit: dict[torch.fx.Node, int] = {}

    for orbit in orbits:
        for node in orbit.sources:
            node_to_out_orbit[node] = orbit.color

        for node in orbit.sinks:
            node_to_inp_orbit[node] = orbit.color

        for node in orbit.non_border:
            node_to_inp_orbit[node] = orbit.color
            node_to_out_orbit[node] = orbit.color

    def __node_to_colors(node: torch.fx.Node) -> tuple[int, int]:
        color1 = node_to_inp_orbit.get(node, -1)
        color2 = node_to_out_orbit.get(node, -1)
        return color1, color2

    return __node_to_colors


def make_orbits_get_style_fn(
    orbits: list[Orbit],
    base_get_style_fn: Callable[..., pth.vis.NodeStyle],
    palette: list[str] = PAL_VEGA_CATEGORY20,
    blank_color: str = "#ffffff",
) -> Callable[..., pth.vis.NodeStyle]:
    node_to_orbits_fn = _make_node_to_orbits_fn(orbits, palette, blank_color)
    colors = sorted([orbit.color for orbit in orbits])

    def __get_style(
        *,
        element: str,
        node_meta1: Optional[dict[str, Any]] = None,
        node_meta2: Optional[dict[str, Any]] = None,
        module_dict: Optional[dict[str, torch.nn.Module]] = None,
    ) -> pth.vis.NodeStyle:
        style = base_get_style_fn(
            element=element,
            node_meta1=node_meta1,
            node_meta2=node_meta2,
            module_dict=module_dict,
        )
        if element == "node":
            assert node_meta1 is not None
            o1, o2 = node_to_orbits_fn(node_meta1["node"])
            color1 = _get_orbit_color(o1, colors, palette, blank_color)
            color2 = _get_orbit_color(o2, colors, palette, blank_color)
            assert isinstance(style["label"], str)
            if o1 != o2:
                style["label"] += f"\norbit={o1}/{o2}"
            else:
                style["label"] += f"\norbit={o1}"
            style["fillcolor"] = f"{color1};0.5:{color2}"
            style["gradientangle"] = "270"
        return style

    return __get_style


def vis_module(
    module: torch.nn.Module,
    orbits: list[Orbit],
    input_shapes: Optional[tuple[tuple[int, ...]]] = None,
    get_style_fn: Callable[..., pth.vis.NodeStyle] = pth.vis.get_std_style,
    **kwargs: Any,
) -> graphviz.Digraph:
    get_style_fn = make_orbits_get_style_fn(orbits, get_style_fn)
    return pth.vis.vis_module(
        module, input_shapes=input_shapes, get_style_fn=get_style_fn, **kwargs
    )
