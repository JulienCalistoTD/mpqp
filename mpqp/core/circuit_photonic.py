from __future__ import annotations

from copy import deepcopy
import numpy as np
import numpy.typing as npt
from sympy import Basic, Expr
from typeguard import TypeCheckError
from numbers import Complex
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Type, Union

from mpqp.core.instruction.barrier import Barrier
from mpqp.core.instruction.gates.photonic_gate import PhotonicGate
from mpqp.core.instruction.instruction import Instruction
from mpqp.core.instruction.gates.parametrized_gate import ParametrizedGate
from mpqp.core.instruction.measurement import BasisMeasure, Measure
from mpqp.core.languages import Language
from mpqp.noise.noise_model import NoiseModel
from mpqp.tools.errors import NumberQubitsError
from mpqp.tools.generics import OneOrMany
from mpqp.tools.maths import matrix_eq


class PhotonicCircuit:
    def __init__(
        self,
        data: int | Sequence[Union[Instruction, NoiseModel]],
        *,
        nb_qubits: Optional[int] = None,
        label: Optional[str] = None,
    ):
        self.label = label
        """See parameter description."""
        self.instructions: list[Instruction] = []
        """List of instructions of the circuit."""
        self.noises: list[NoiseModel] = []
        """List of noise models attached to the circuit."""
        self.nb_qubits: int
        """Number of qubits of the circuit."""

        if isinstance(data, int):
            if data < 0:
                raise TypeCheckError(
                    f"The data passed to PhotonicCircuit is a negative int ({data}), "
                    "this does not make sense."
                )
            self.nb_qubits = data
        else:
            if nb_qubits is None:
                if len(data) == 0:
                    self.nb_qubits = 0
                else:
                    connections: set[int] = set.union(
                        *(item.connections() for item in data)
                    )
                    self.nb_qubits = max(connections) + 1
            else:
                self.nb_qubits = nb_qubits
            self.add(list(map(deepcopy, data)))

    def add(self, components: OneOrMany[Instruction | NoiseModel]):
        """# TODO: DOC """

        if isinstance(components, Iterable):
            for comp in components:
                self.add(comp)
            return

        if any(conn >= self.nb_qubits for conn in components.connections()):
            component_type = (
                "Instruction" if isinstance(components, Instruction) else "Noise model"
            )
            raise NumberQubitsError(
                f"{component_type} {type(components)}'s connections "
                f"({components.connections()}) are not compatible with circuit"
                f" size ({self.nb_qubits})."
            )

        if isinstance(components, BasisMeasure):
            pass
        if isinstance(components, Barrier):
            components.size = self.nb_qubits
        if isinstance(components, NoiseModel):
            self.noises.append(components)

        if isinstance(components, PhotonicGate):
            self.instructions.append(components)
        else:
            raise ValueError(f"Unknown component type ({ repr(components) }).")

    def to_other_language(
        self, language: Language = Language.PERCEVAL, cirq_proc_id: Optional[str] = None
    ):
        """# TODO: DOC"""
        if language == Language.PERCEVAL:
            import perceval as pcvl

            perceval_circuit = pcvl.Circuit(self.nb_qubits)
            for instruction in self.instructions:
                if isinstance(instruction, PhotonicGate):
                    if len(instruction.targets) == 1:
                        targets = (instruction.targets[0],)
                    elif len(instruction.targets) == 2:
                        targets = (instruction.targets[0], instruction.targets[1])
                    elif len(instruction.targets) == 3:
                        targets = (
                            instruction.targets[0],
                            instruction.targets[1],
                            instruction.targets[2],
                        )
                    else:
                        raise ValueError("Invalid number of elements in the list")

                    perceval_circuit.add(
                        targets, instruction.to_other_language(language)
                    )
            return perceval_circuit

    def __str__(self) -> str:
        from perceval.rendering.format import Format
        from perceval.rendering.pdisplay import _pdisplay  # type: ignore

        pcvl_circ = self.to_other_language(Language.PERCEVAL)
        if TYPE_CHECKING:
            from perceval import Circuit

            assert isinstance(pcvl_circ, Circuit)
        output = str(_pdisplay(pcvl_circ, output_format=Format.TEXT))
        if TYPE_CHECKING:
            assert isinstance(output, str)
        if len(self.noises) != 0:
            output += "\nNoiseModel:\n    " + "\n    ".join(
                str(noise) for noise in self.noises
            )
        return output

    def append(self, other: PhotonicCircuit, qubits_offset: int = 0) -> None:
        """# TODO: DOC"""
        assert isinstance(other, PhotonicCircuit)
        if self.nb_qubits < other.nb_qubits:
            raise NumberQubitsError(
                "Size of the circuit to be appended is greater than the size of"
                " this circuit"
            )
        if qubits_offset + other.nb_qubits > self.nb_qubits:
            raise NumberQubitsError(
                "Size of the circuit to be appended is too large given the"
                " index and the size of this circuit"
            )

        for inst in deepcopy(other.instructions):
            inst.targets = [qubit + qubits_offset for qubit in inst.targets]
            # if isinstance(inst, ControlledGate):
            #    inst.controls = [qubit + qubits_offset for qubit in inst.controls]
            if isinstance(inst, BasisMeasure):
                if not inst.user_set_c_targets:
                    inst.c_targets = None

            self.add(inst)

    def __iadd__(self, other: PhotonicCircuit) -> PhotonicCircuit:
        self.append(other)
        return self

    def __add__(self, other: PhotonicCircuit) -> PhotonicCircuit:
        res = deepcopy(self)
        res += other
        return res

    def tensor(self, other: PhotonicCircuit) -> PhotonicCircuit:
        """# TODO: DOC
        Computes the tensor product of this circuit with the one in parameter.

        In the circuit notation, the upper part of the output circuit will
        correspond to the first circuit, while the bottom part correspond to the
        one in parameter.

        Args:
            other: PhotonicCircuit being the second operand of the tensor product with
                this circuit.

        Returns:
            The PhotonicCircuit resulting from the tensor product of this circuit with
            the one in parameter.

        Args:
            other: PhotonicCircuit being the second operand of the tensor product with this circuit.

        Returns:
            The PhotonicCircuit resulting from the tensor product of this circuit with the one in parameter.

        Example:
            >>> c1 = PhotonicCircuit([CNOT(0,1),CNOT(1,2)])
            >>> c2 = PhotonicCircuit([X(1),CNOT(1,2)])
            >>> print(c1.tensor(c2))  # doctest: +NORMALIZE_WHITESPACE
            q_0: ──■───────
                 ┌─┴─┐
            q_1: ┤ X ├──■──
                 └───┘┌─┴─┐
            q_2: ─────┤ X ├
                      └───┘
            q_3: ──────────
                 ┌───┐
            q_4: ┤ X ├──■──
                 └───┘┌─┴─┐
            q_5: ─────┤ X ├
                      └───┘

        """
        res = deepcopy(self)
        res.nb_qubits += other.nb_qubits
        res.append(other, qubits_offset=self.nb_qubits)
        return res

    def __matmul__(self, other: PhotonicCircuit) -> PhotonicCircuit:
        return self.tensor(other)

    def display(self, output: str = "mpl"):
        r"""# TODO: DOC """
        from perceval.rendering.format import Format
        from perceval.rendering.pdisplay import pdisplay

        pcvl_circ = self.to_other_language(Language.PERCEVAL)
        if TYPE_CHECKING:
            from perceval import Circuit

            assert isinstance(pcvl_circ, Circuit)
        if output == "mpl":
            format = Format.MPLOT
        elif output == "html":
            format = Format.HTML
        elif output == "text":
            format = Format.TEXT
        elif output == "latex":
            format = Format.LATEX
        else:
            raise ValueError(f"Unknown output format ({output}).")
        return pdisplay(pcvl_circ, output_format=format)

    def size(self) -> int:
        """# TODO: DOC """
        return self.nb_qubits

    def depth(self) -> int:
        """# TODO: DOC
        Computes the depth of the circuit.

        Returns:
            Depth of the circuit.

        Examples:
            >>> PhotonicCircuit([CNOT(0, 1), CNOT(1, 2), CNOT(0, 1), X(2)]).depth()
            3
            >>> PhotonicCircuit([CNOT(0, 1), CNOT(1, 2), CNOT(0, 1), Barrier(), X(2)]).depth()
            4

        """
        if len(self) == 0:
            return 0

        nb_qubits = self.nb_qubits
        instructions = self.without_measurements().instructions
        layers = np.zeros((nb_qubits, self.count_gates()), dtype=bool)

        current_layer = 0
        last_barrier = 0
        for instr in instructions:
            if isinstance(instr, Barrier):
                last_barrier = current_layer
                current_layer += 1
                continue
            conns = list(instr.connections())
            if any(layers[conns, current_layer]):
                current_layer += 1
            fitting_layer_index = current_layer
            for index in range(current_layer, last_barrier - 1, -1):
                if any(layers[conns, index]):
                    fitting_layer_index = index + 1
                    break
            layers[conns, fitting_layer_index] = [True] * len(conns)

        return current_layer + 1

    def __len__(self) -> int:
        """# TODO: DOCReturns the number of instructions added to this circuit.

        Returns:
            An integer representing the number of instructions in this circuit.

        Example:
            >>> c1 = PhotonicCircuit([CNOT(0,1), CNOT(1,2), X(1), CNOT(1,2)])
            >>> len(c1)
            4

        """
        return len(self.instructions)

    def is_equivalent(self, circuit: PhotonicCircuit) -> bool:
        """# TODO: DOCWhether the circuit in parameter is equivalent to this circuit, in
        terms of gates, but not measurements.

        Depending on the definition of the gates of the circuit, several methods
        could be used to do it in an optimized way.

        Args:
            circuit: The circuit for which we want to know if it is equivalent
                to this circuit.

        Returns:
            ``True`` if the circuit in parameter is equivalent to this circuit

        Example:
            >>> c1 = PhotonicCircuit([H(0), H(0)])
            >>> c2 = PhotonicCircuit([Rx(0, 0)])
            >>> c1.is_equivalent(c2)
            True

        3M-TODO: will only work once the circuit.to_matrix is implemented
         Also take into account Noise in the equivalence verification
        """
        return matrix_eq(self.to_matrix(), circuit.to_matrix())

    def optimize(self, criteria: Optional[OneOrMany[str]] = None) -> PhotonicCircuit:
        """# TODO: DOC
        Optimize the circuit to satisfy some criteria (depth, number of
        qubits, gate restriction) in parameter.

        Args:
            criteria: String, or list of strings, regrouping the criteria of optimization of the circuit.

        Returns:
            the optimized PhotonicCircuit

        Examples:
            >>>
            >>>
            >>>

        # 6M-TODO implement, example and test
        """
        # ideas: a circuit can be optimized
        # - to reduce the depth of the circuit (combine gates, simplify some sequences)
        # - according to a given topology or qubits connectivity map
        # - to avoid the use of some gates (imperfect or more noisy)
        # - to avoid multi-qubit gates
        ...

    def to_matrix(self) -> npt.NDArray[np.complex64]:
        """# TODO: DOC
        Compute the unitary matrix associated to this circuit.

        Returns:
            a unitary matrix representing this circuit

        Examples:
            >>> c = PhotonicCircuit([H(0), CNOT(0,1)])
            >>> c.to_matrix()
            array([[ 0.70710678,  0.        ,  0.70710678,  0.        ],
                   [ 0.        ,  0.70710678,  0.        ,  0.70710678],
                   [ 0.        ,  0.70710678,  0.        , -0.70710678],
                   [ 0.70710678,  0.        , -0.70710678,  0.        ]])

        # 3M-TODO implement and double check examples and test:
        the idea is to compute the tensor product of the matrices associated
        with the gates of the circuit in a clever way (to minimize the number of
        multiplications) and then return the big matrix
        """
        ...

    def inverse(self) -> PhotonicCircuit:
        """# TODO: DOC
        Generate the inverse (dagger) of this circuit.

        Returns:
            The inverse circuit.

        Examples:
            >>> c1 = PhotonicCircuit([H(0), CNOT(0,1)])
            >>> print(c1)  # doctest: +NORMALIZE_WHITESPACE
                 ┌───┐
            q_0: ┤ H ├──■──
                 └───┘┌─┴─┐
            q_1: ─────┤ X ├
                      └───┘
            >>> print(c1.inverse())  # doctest: +NORMALIZE_WHITESPACE
                      ┌───┐
            q_0: ──■──┤ H ├
                 ┌─┴─┐└───┘
            q_1: ┤ X ├─────
                 └───┘
            >>> c2 = PhotonicCircuit([S(0), CZ(0,1), H(1), Ry(4.56, 1)])
            >>> print(c2)  # doctest: +NORMALIZE_WHITESPACE
                 ┌───┐
            q_0: ┤ S ├─■──────────────────
                 └───┘ │ ┌───┐┌──────────┐
            q_1: ──────■─┤ H ├┤ Ry(4.56) ├
                         └───┘└──────────┘
            >>> print(c2.inverse())  # doctest: +NORMALIZE_WHITESPACE
                                     ┌───┐
            q_0: ──────────────────■─┤ S ├
                 ┌──────────┐┌───┐ │ └───┘
            q_1: ┤ Ry(4.56) ├┤ H ├─■──────
                 └──────────┘└───┘

        # 3M-TODO implement, test, fill second example
        The inverse could be computed in several ways, depending on the
        definition of the circuit. One can inverse each gate in the circuit, or
        take the global unitary of the gate and inverse it.
        """
        dagger = PhotonicCircuit(self.nb_qubits)
        for instr in reversed(self.instructions):
            dagger.add(instr)
        return dagger

    def to_gate(self):
        """# TODO: DOC
        Generate a gate from this entire circuit.

        Returns:
            A gate representing this circuit.

        # 3M-TODO check implementation, example and test, this will only work
           when circuit.to_matrix() will be implemented
        """
        pass

    @classmethod
    def initializer(cls, state: npt.NDArray[np.complex64]):
        """# TODO: DOC
        Initialize this circuit at a given state, given in parameter.
        This will imply adding gates at the beginning of the circuit.

        Args:
            state: StateVector modeling the state for initializing the circuit.

        Returns:
            A copy of the input circuit with additional instructions added
            before-hand to generate the right initial state.

        # 3M-TODO : to implement --> a first sort term way could be to reuse the
        # qiskit QuantumCircuit feature qc.initialize()
        """
        pass

    def count_gates(self, gate: Optional[Type[PhotonicGate]] = None) -> int:
        """# TODO: DOC
        Returns the number of gates contained in the circuit. If a specific
        gate is given in the ``gate`` arg, it returns the number of occurrences
        of this gate.

        Args:
            gate: The gate for which we want to know its occurrence in this
                circuit.

        Returns:
            The number of gates (eventually of a specific type) contained in the
            circuit.

        Examples:
            >>> circuit = PhotonicCircuit(
            ...     [X(0), Y(1), Z(2), CNOT(0, 1), SWAP(0, 1), CZ(1, 2), X(2), X(1), X(0)]
            ... )
            >>> circuit.count_gates()
            9
            >>> circuit.count_gates(X)
            4
            >>> circuit.count_gates(Ry)
            0

        """
        filter2 = PhotonicGate if gate is None else gate
        return len([inst for inst in self.instructions if isinstance(inst, filter2)])

    def get_measurements(self) -> list[Measure]:
        """# TODO: DOC
        Returns all the measurements present in this circuit.

        Returns:
            The list of all measurements present in the circuit.

        Example:
            >>> circuit = PhotonicCircuit([
            ...     BasisMeasure([0, 1], shots=1000),
            ...     ExpectationMeasure([1], Observable(np.identity(2)), shots=1000)
            ... ])
            >>> circuit.get_measurements()  # doctest: +NORMALIZE_WHITESPACE
            [BasisMeasure([0, 1], shots=1000),
            ExpectationMeasure([1], Observable(array([[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]], dtype=complex64)), shots=1000)]

        """
        return [inst for inst in self.instructions if isinstance(inst, Measure)]

    def without_measurements(self) -> PhotonicCircuit:
        """# TODO: DOC
        Provides a copy of this circuit with all the measurements removed.

        Returns:
            A copy of this circuit with all the measurements removed.

        Example:
            >>> circuit = PhotonicCircuit([X(0), CNOT(0, 1), BasisMeasure([0, 1], shots=100)])
            >>> print(circuit)  # doctest: +NORMALIZE_WHITESPACE
                 ┌───┐     ┌─┐
            q_0: ┤ X ├──■──┤M├───
                 └───┘┌─┴─┐└╥┘┌─┐
            q_1: ─────┤ X ├─╫─┤M├
                      └───┘ ║ └╥┘
            c: 2/═══════════╩══╩═
                            0  1
            >>> print(circuit.without_measurements())  # doctest: +NORMALIZE_WHITESPACE
                 ┌───┐
            q_0: ┤ X ├──■──
                 └───┘┌─┴─┐
            q_1: ─────┤ X ├
                      └───┘

        """
        new_circuit = PhotonicCircuit(self.nb_qubits)
        new_circuit.instructions = [
            inst for inst in self.instructions if not isinstance(inst, Measure)
        ]

        return new_circuit

    def without_noises(self) -> PhotonicCircuit:
        """# TODO: DOC
        Provides a copy of this circuit with all the noise models removed.

        Returns:
            A copy of this circuit with all the noise models removed.

        Example:
            >>> circuit = PhotonicCircuit(2)
            >>> circuit.add([CNOT(0, 1), Depolarizing(prob=0.4, targets=[0, 1]), BasisMeasure([0, 1], shots=100)])
            >>> print(circuit)  # doctest: +NORMALIZE_WHITESPACE
                      ┌─┐
            q_0: ──■──┤M├───
                 ┌─┴─┐└╥┘┌─┐
            q_1: ┤ X ├─╫─┤M├
                 └───┘ ║ └╥┘
            c: 2/══════╩══╩═
                       0  1
            NoiseModel: Depolarizing(0.4, [0, 1], 1)
            >>> print(circuit.without_noises())  # doctest: +NORMALIZE_WHITESPACE
                      ┌─┐
            q_0: ──■──┤M├───
                 ┌─┴─┐└╥┘┌─┐
            q_1: ┤ X ├─╫─┤M├
                 └───┘ ║ └╥┘
            c: 2/══════╩══╩═
                       0  1

        """
        new_circuit = deepcopy(self)
        new_circuit.noises = []
        return new_circuit

    def to_qasm2(self):
        """# TODO: DOC
        Converts this circuit to the corresponding OpenQASM 2 code.

        For now, we use an intermediate conversion to a Qiskit
        ``QuantumCircuit``.

        Returns:
            A string representing the OpenQASM2 code corresponding to this
            circuit.

        # 3M-TODO
        """
        pass

    def to_qasm3(self):
        """# TODO: DOC
        Converts this circuit to the corresponding OpenQASM 3 code.

        For now, we use an intermediate conversion to OpenQASM 2, and then a
        converter from 2 to 3.

        Returns:
            A string representing the OpenQASM3 code corresponding to this
            circuit.

        # 3M-TODO
        """
        pass

    def subs(
        self, values: dict[Expr | str, Complex], remove_symbolic: bool = False
    ) -> PhotonicCircuit:
        r"""#TODO: doc
        Substitute the parameters of the circuit with complex values.
        Optionally also remove all symbolic variables such as `\pi` (needed for
        example for circuit execution).

        Since we use ``sympy`` for gates' parameters, ``values`` can in fact be
        anything the ``subs`` method from ``sympy`` would accept.

        Args:
            values: Mapping between the variables and the replacing values.
            remove_symbolic: If symbolic values should be replaced by their
                numeric counterpart.

        Returns:
            The circuit with the replaced parameters.

        """
        return PhotonicCircuit(
            data=[inst.subs(values, remove_symbolic) for inst in self.instructions]
            + self.noises,  # 3M-TODO: modify this line when noise will be parameterized, to substitute, like we do for inst
            nb_qubits=self.nb_qubits,
            label=self.label,
        )

    def __repr__(self) -> str:
        instructions_repr = ", ".join(repr(instr) for instr in self.instructions)
        instructions_repr = instructions_repr.replace("[", "").replace("]", "")

        if self.noises:
            noise_repr = ", ".join(map(repr, self.noises))
            return f'PhotonicCircuit([{instructions_repr}, {noise_repr}], nb_qubits={self.nb_qubits}, label="{self.label}")'
        else:
            return f'PhotonicCircuit([{instructions_repr}], nb_qubits={self.nb_qubits}, label="{self.label}")'

    def variables(self) -> set[Basic]:
        """#TODO: doc
        Returns all the parameters involved in this circuit.

        Returns:
            All the parameters of the circuit.
        """
        params: set[Basic] = set()
        for inst in self.instructions:
            if isinstance(inst, ParametrizedGate):
                for param in inst.parameters:
                    if isinstance(param, Expr):
                        params.update(param.free_symbols)
        return params
