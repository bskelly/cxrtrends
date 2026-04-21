from dataclasses import dataclass
from fractions import Fraction
from typing import Union


@dataclass
class WeightCluster:
    """Group of tasks that split a benchmark weight."""

    weight: Union[int, Fraction]
    tasks: list[type]
