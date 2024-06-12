from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt
from typeguard import typechecked

if TYPE_CHECKING:
    from qiskit.circuit import Parameter

from mpqp.core.instruction.gates.gate import Gate
from mpqp.core.languages import Language
from mpqp.tools.generics import Matrix


@typechecked
class PhotonicGate(Gate, ABC):
    """Abstract base class for photonic gates."""

    def __init__(self, targets: list[int], label: Optional[str] = None):
        super().__init__(targets, label)

    matrix: npt.NDArray[np.complex64]

    def to_matrix(self) -> Matrix:
        return self.matrix


class BeamSplitter(PhotonicGate):
    def __init__(
        self,
        target1: int,
        target2: int,
        label: Optional[str] = None,
        theta: float = np.pi / 2,
        phi_tl: int = 0,
        phi_bl: int = 0,
        phi_tr: int = 0,
        phi_br: int = 0,
    ):
        super().__init__([target1, target2], label)
        self.theta = theta
        self.phi_tl = phi_tl
        self.phi_bl = phi_bl
        self.phi_tr = phi_tr
        self.phi_br = phi_br
        self.matrix = np.array(
            [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]]
        )

    def to_other_language(
        self,
        language: Language = Language.PERCEVAL,
        qiskit_parameters: Optional[set[Parameter]] = None,
    ):
        from perceval.components import BS

        if language == Language.PERCEVAL:
            return BS(self.theta, self.phi_tl, self.phi_bl, self.phi_tr, self.phi_br)
        else:
            raise NotImplementedError(f"Error: {language} is not supported")


class PhaseShifter(PhotonicGate):
    def __init__(self, target: int, phi: float, label: Optional[str] = None):
        self.phi = phi
        super().__init__([target], label)
        self.matrix = np.array([[1, 0], [0, np.exp(1j * self.phi)]])

    def to_other_language(
        self,
        language: Language = Language.PERCEVAL,
        qiskit_parameters: Optional[set[Parameter]] = None,
    ):
        from perceval.components import PS

        if language == Language.PERCEVAL:
            return PS(self.phi)
        else:
            raise NotImplementedError(f"Error: {language} is not supported")

    
