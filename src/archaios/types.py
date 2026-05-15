from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union
import numpy as np

Grid = np.ndarray


class SegmentationMode(str, Enum):
    COLOR = "COLOR"
    SPATIAL = "SPATIAL"


@dataclass(frozen=True)
class SegmentationProfile:
    mode: SegmentationMode = SegmentationMode.COLOR
    connectivity: int = 4
    background: Optional[int] = None
    background_strategy: str = "auto"

    def label(self) -> str:
        bg = "auto" if self.background is None else str(self.background)
        return f"{self.mode.value}+{self.connectivity}+bg={bg}"


@dataclass(frozen=True)
class ArcObject:
    id: str
    color: int
    pixels: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    area: int
    centroid: tuple[float, float]
    touches_border: bool
    width: int
    height: int
    shape_signature: tuple[tuple[int, int], ...]


@dataclass
class Invariants:
    preserved: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    candidate_transform_families: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    profile_label: str = ""


@dataclass(frozen=True)
class Selector:
    kind: str
    value: Optional[Union[int, str]] = None

    @staticmethod
    def all() -> "Selector":
        return Selector("ALL")

    @staticmethod
    def color(color: int) -> "Selector":
        return Selector("COLOR", color)

    @staticmethod
    def size(size: int) -> "Selector":
        return Selector("SIZE", size)

    @staticmethod
    def largest() -> "Selector":
        return Selector("LARGEST")

    @staticmethod
    def smallest() -> "Selector":
        return Selector("SMALLEST")

    @staticmethod
    def touching_border() -> "Selector":
        return Selector("TOUCHING_BORDER")

    @staticmethod
    def not_touching_border() -> "Selector":
        return Selector("NOT_TOUCHING_BORDER")

    @staticmethod
    def position(position: str) -> "Selector":
        if position not in {"left", "right", "top", "bottom"}:
            raise ValueError("position selector must be one of: left, right, top, bottom")
        return Selector("POSITION", position)

    @staticmethod
    def shape(shape: str) -> "Selector":
        allowed = {"square", "rectangle", "line_horizontal", "line_vertical", "single_pixel"}
        if shape not in allowed:
            raise ValueError(f"shape selector must be one of: {sorted(allowed)}")
        return Selector("SHAPE", shape)

    def to_dsl(self) -> str:
        if self.kind == "ALL":
            return "ALL"
        if self.kind == "COLOR":
            return f"OBJECTS(color={self.value})"
        if self.kind == "SIZE":
            return f"OBJECTS(size={self.value})"
        if self.kind == "LARGEST":
            return "SELECT_LARGEST(OBJECTS())"
        if self.kind == "SMALLEST":
            return "SELECT_SMALLEST(OBJECTS())"
        if self.kind == "TOUCHING_BORDER":
            return "OBJECTS(touches_border=true)"
        if self.kind == "NOT_TOUCHING_BORDER":
            return "OBJECTS(touches_border=false)"
        if self.kind == "POSITION":
            return f"OBJECTS(position={self.value})"
        if self.kind == "SHAPE":
            return f"OBJECTS(shape={self.value})"
        return f"UNKNOWN_SELECTOR({self.kind},{self.value})"


@dataclass(frozen=True)
class ShiftProgram:
    selector: Selector
    dx: int
    dy: int

    def to_dsl(self) -> str:
        return f"SHIFT({self.selector.to_dsl()}, dx={self.dx}, dy={self.dy})"


@dataclass(frozen=True)
class RecolorProgram:
    selector: Selector
    color_map: dict[int, int]

    def to_dsl(self) -> str:
        mapping = ", ".join(f"{k}->{v}" for k, v in sorted(self.color_map.items()) if k != v)
        return f"RECOLOR({self.selector.to_dsl()}, {{{mapping}}})"


@dataclass(frozen=True)
class DeleteProgram:
    selector: Selector

    def to_dsl(self) -> str:
        return f"DELETE({self.selector.to_dsl()})"


@dataclass(frozen=True)
class FillBBoxProgram:
    selector: Selector
    color: int

    def to_dsl(self) -> str:
        return f"FILL_BBOX({self.selector.to_dsl()}, color={self.color})"


@dataclass(frozen=True)
class SeqProgram:
    first: "Program"
    second: "Program"

    def to_dsl(self) -> str:
        return f"SEQ({self.first.to_dsl()}, {self.second.to_dsl()})"


Program = Union[ShiftProgram, RecolorProgram, DeleteProgram, FillBBoxProgram, SeqProgram]


@dataclass
class Hypothesis:
    program: Program
    generator: str
    confidence: float
    train_match: str
    match_rate: float
    notes: list[str] = field(default_factory=list)

    @property
    def program_dsl(self) -> str:
        return self.program.to_dsl()
