from sympy import Symbol
import sympy.physics.quantum.operator as qmoperator
from responsefun.adcc_properties import available_operators


available_operators_symb = [Symbol(op) for op in available_operators]
for op_type, tup in available_operators.items():
    if tup[1] not in [0, 1, 2]:
        raise ValueError(
                f"An unknown symmetry was specified for the {op_type} operator. "
                "Only the following symmetries are allowed:\n"
                "0: no symmetry assumed, 1: hermitian, 2: anti-hermitian"
        )


class ResponseOperator(qmoperator.Operator):
    """
    Base class for (state-to-state) modified transition moments and response vectors.
    """
    def __init__(self, comp):
        """
        Parameters
        ----------
        comp: str
            Cartesian component.
        """
        self._comp = comp

    @property
    def comp(self):
        return self._comp

    def _print_contents(self, printer):
        return "{}_{{{}}}".format(self.__class__.__name__, self._comp)


class MTM(ResponseOperator):
    def __init__(self, comp, op_type):
        super().__init__(comp)
        assert op_type in available_operators or op_type in available_operators_symb
        self._op_type = op_type
        if isinstance(op_type, Symbol):
            op_type_str = str(op_type)
        else:
            op_type_str = op_type
        self._symmetry = available_operators[op_type_str][1]
        self._dim = available_operators[op_type_str][2]
        if isinstance(comp, Symbol):
            comp_str = str(comp)
        else:
            comp_str = comp
        if len(comp_str) != self._dim:
            raise ValueError(
                    f"The operator is {self._dim}-dimensional, but {len(comp_str)} components were specified."
            )
        
    @property
    def op_type(self):
        return self._op_type

    @property
    def symmetry(self):
        return self._symmetry

    @property
    def dim(self):
        return self._dim

    def _print_contents(self, printer):
        if self._op_type == "electric":
            return "F_{{{}}}".format(self._comp)
        else:
            return "F({})_{{{}}}".format(available_operators[self._op_type][0], self._comp)

    def _print_contents_latex(self, printer):
        if self._op_type == "electric":
            return "F_{{{}}}".format(self._comp)
        else:
            return "F({})_{{{}}}".format(available_operators[self._op_type][0], self._comp)


class S2S_MTM(ResponseOperator):
    def __init__(self, comp, op_type):
        super().__init__(comp)
        assert op_type in available_operators or op_type in available_operators_symb
        self._op_type = op_type
        if isinstance(op_type, Symbol):
            op_type_str = str(op_type)
        else:
            op_type_str = op_type
        self._symmetry = available_operators[op_type_str][1]
        self._dim = available_operators[op_type_str][2]
        if isinstance(comp, Symbol):
            comp_str = str(comp)
        else:
            comp_str = comp
        if len(comp_str) != self._dim:
            raise ValueError(
                    f"The operator is {self._dim}-dimensional, but {len(comp_str)} components were specified."
            )

    @property
    def op_type(self):
        return self._op_type

    @property
    def symmetry(self):
        return self._symmetry

    @property
    def dim(self):
        return self._dim

    def _print_contents(self, printer):
        if self._op_type == "electric":
            return "B_{{{}}}".format(self._comp)
        else:
            return "B({})_{{{}}}".format(available_operators[self._op_type][0], self._comp)

    def _print_contents_latex(self, printer):
        if self._op_type == "electric":
            return "B_{{{}}}".format(self._comp)
        else:
            return "B({})_{{{}}}".format(available_operators[self._op_type][0], self._comp)


class ResponseVector(ResponseOperator):
    def __init__(self, comp, no=None, mtm_type=None, symmetry=None):
        if mtm_type:
            assert mtm_type in ["MTM", "S2S_MTM"]
        if symmetry:
            assert symmetry in [0, 1, 2]
        super().__init__(comp)
        self._no = no
        self._mtm_type = mtm_type
        self._symmetry = symmetry

    @property
    def no(self):
        return self._no

    @property
    def mtm_type(self):
        return self._mtm_type

    @property
    def symmetry(self):
        return self._symmetry

    def _print_contents(self, printer):
        return "X_{{{}, {}}}".format(self._comp, self._no)

    def _print_contents_latex(self, printer):
        return "X_{{{}, {}}}".format(self._comp, self._no)


class DipoleOperator(ResponseOperator):
    def __init__(self, comp, op_type):
        super().__init__(comp)
        assert op_type in available_operators or op_type in available_operators_symb
        self._op_type = op_type
        if isinstance(op_type, Symbol):
            op_type_str = str(op_type)
        else:
            op_type_str = op_type
        self._symmetry = available_operators[op_type_str][1]
        self._dim = available_operators[op_type_str][2]
        if isinstance(comp, Symbol):
            comp_str = str(comp)
        else:
            comp_str = comp
        if len(comp_str) != self._dim:
            raise ValueError(
                    f"The operator is {self._dim}-dimensional, but {len(comp_str)} components were specified."
            )

    @property
    def op_type(self):
        return self._op_type

    @property
    def symmetry(self):
        return self._symmetry

    @property
    def dim(self):
        return self._dim

    def _print_contents(self, printer):
        return r"{}_{{{}}}".format(available_operators[self._op_type][0], self._comp)

    def _print_contents_latex(self, printer):
        return r"{}_{{{}}}".format(available_operators[self._op_type][0], self._comp)


class DipoleMoment(Symbol):
    def __new__(self, comp, from_state, to_state, op_type, **assumptions):
        assert type(comp) == str
        assert type(from_state) == str
        assert type(to_state) == str
        assert op_type in available_operators
        name = r"{}_{{{}}}^{{{}}}".format(available_operators[op_type][0], comp, from_state+to_state)
        obj = Symbol.__new__(self, name, **assumptions)
        obj._comp = comp
        obj._from_state = from_state
        obj._to_state = to_state
        obj._op_type = op_type
        obj._symmetry = available_operators[op_type][1]
        obj._dim = available_operators[op_type][2]
        if len(comp) != obj._dim:
            raise ValueError(
                    f"The operator is {obj._dim}-dimensional, but {len(comp)} components were specified."
            )
        return obj

    @property
    def comp(self):
        return self._comp

    @property
    def from_state(self):
        return self._from_state
    
    @property
    def to_state(self):
        return self._to_state

    @property
    def op_type(self):
        return self._op_type

    @property
    def symmetry(self):
        return self._symmetry

    @property
    def dim(self):
        return self._dim


class TransitionFrequency(Symbol):
    def __new__(self, state, **assumptions):
        assert type(state) == str
        name = r"w_{{{}}}".format(state)
        obj = Symbol.__new__(self, name, **assumptions)
        obj._state = state
        return obj

    @property
    def state(self):
        return self._state


class LeviCivita(qmoperator.Operator):
    def _print_contents(self, printer):
        return r"\epsilon_{ABC}"

    def _print_contents_latex(self, printer):
        return r"\epsilon_{ABC}"


