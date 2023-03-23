from sympy import Symbol
import sympy.physics.quantum.operator as qmoperator
from responsefun.AdccProperties import available_operators


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
        if isinstance(comp, Symbol):
            self._comp = str(comp)
        else:
            assert isinstance(comp, str)
            self._comp = comp

    @property
    def comp(self):
        return self._comp

    def _print_contents(self, printer):
        return "{}_{{{}}}".format(self.__class__.__name__, self._comp)


class MTM(ResponseOperator):
    def __init__(self, comp, op_type):
        super().__init__(comp)
        if isinstance(op_type, Symbol):
            self._op_type = str(op_type)
        else:
            assert isinstance(op_type, str)
            self._op_type = op_type
        assert self._op_type in available_operators
        self._symmetry = available_operators[self._op_type][1]
        self._dim = available_operators[self._op_type][2]
        if len(self._comp) != self._dim:
            raise ValueError(
                    f"The operator is {self._dim}-dimensional, but {len(self._comp)} components were specified."
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
        if isinstance(op_type, Symbol):
            self._op_type = str(op_type)
        else:
            assert isinstance(op_type, str)
            self._op_type = op_type
        assert self._op_type in available_operators

        self._symmetry = available_operators[self._op_type][1]
        self._dim = available_operators[self._op_type][2]
        if len(self._comp) != self._dim:
            raise ValueError(
                    f"The operator is {self._dim}-dimensional, but {len(self._comp)} components were specified."
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


class OneParticleOperator(ResponseOperator):
    def __init__(self, comp, op_type):
        super().__init__(comp)
        if isinstance(op_type, Symbol):
            self._op_type = str(op_type)
        else:
            assert isinstance(op_type, str)
            self._op_type = op_type
        assert self._op_type in available_operators

        self._symmetry = available_operators[self._op_type][1]
        self._dim = available_operators[self._op_type][2]
        if len(self._comp) != self._dim:
            raise ValueError(
                    f"The operator is {self._dim}-dimensional, but {len(self._comp)} components were specified."
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


class Moment(Symbol):
    def __new__(self, comp, from_state, to_state, op_type, **assumptions):
        assert isinstance(comp, str)
        assert isinstance(from_state, Symbol)
        assert isinstance(to_state, Symbol)
        assert op_type in available_operators
        name = r"{}_{{{}}}^{{{}}}".format(available_operators[op_type][0], comp, str(from_state)+str(to_state))
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
        assert isinstance(state, Symbol)
        name = r"w_{{{}}}".format(str(state))
        obj = Symbol.__new__(self, name, **assumptions)
        obj._state = state
        return obj

    @property
    def state(self):
        return self._state
