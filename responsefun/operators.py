import sympy.physics.quantum.operator as qmoperator
from sympy import Symbol
from sympy.logic.boolalg import Boolean
from sympy.physics.quantum.operator import Operator

from responsefun.AdccProperties import Symmetry, get_operator_by_name


# ADC matrix (for internal use)
M = Operator("M")


class GeneralOperator(qmoperator.Operator):
    """Base class for (state-to-state) modified transition moments and response vectors."""

    def __new__(cls, comp, *args, **kwargs):
        """
        Parameters
        ----------
        comp: str
            Cartesian component.
        """
        obj = qmoperator.Operator.__new__(cls, comp, *args, **kwargs)
        if isinstance(comp, Symbol):
            obj._comp = str(comp)
        else:
            assert isinstance(comp, str)
            obj._comp = comp
        return obj

    @property
    def comp(self):
        return self._comp

    def _print_contents(self, printer):
        return "{}_{}".format(self.__class__.__name__, self._comp)


class PropertyOperator(GeneralOperator):
    def __new__(cls, comp, op_type, *args, **kwargs):
        obj = GeneralOperator.__new__(cls, comp, op_type, *args, **kwargs)
        if isinstance(op_type, Symbol):
            op_type = str(op_type)
        assert isinstance(op_type, str)
        
        obj._operator = get_operator_by_name(op_type)
        
        if len(obj._comp) != obj._operator.dim:
            raise ValueError(
                f"The operator is {obj._operator.dim}-dimensional, but {len(obj._comp)} "
                "components were specified."
            )
        return obj

    @property
    def op_type(self):
        return self._operator.name

    @property
    def symmetry(self):
        return self._operator.symmetry

    @property
    def dim(self):
        return self._dim


class OneParticleOperator(PropertyOperator):
    def __new__(cls, comp, op_type, shifted):
        obj = PropertyOperator.__new__(cls, comp, op_type, shifted)
        assert isinstance(shifted, bool) or isinstance(shifted, Boolean)
        obj._shifted = shifted
        return obj

    def copy_with_new_shifted(self, shifted):
        return OneParticleOperator(self.comp, self.op_type, shifted)

    @property
    def shifted(self):
        return self._shifted

    def _print_contents(self, printer):
        op = self._operator.symbol
        if self.shifted:
            return "{}_{}_bar".format(op, self.comp)
        else:
            return "{}_{}".format(op, self.comp)

    def _print_contents_latex(self, printer):
        op = self._operator.symbol
        if len(op) > 1:
            op = "\\" + op
        if self.shifted:
            return "\\hat{{\\overline{{{}}}}}_{{{}}}".format(op, self.comp)
        else:
            return "\\hat{{{}}}_{{{}}}".format(op, self.comp)


class MTM(PropertyOperator):
    def _print_contents(self, printer):
        op = self._operator.symbol
        return "F({})_{}".format(op, self.comp)

    def _print_contents_latex(self, printer):
        op = self._operator.symbol
        if len(op) > 1:
            op = "\\" + op
        return "F({})_{{{}}}".format(op, self.comp)


class S2S_MTM(PropertyOperator):
    def _print_contents(self, printer):
        op = self._operator.symbol
        return "B({})_{}".format(op, self.comp)

    def _print_contents_latex(self, printer):
        op = self._operator.symbol
        if len(op) > 1:
            op = "\\" + op
        return "B({})_{{{}}}".format(op, self.comp)


class ResponseVector(GeneralOperator):
    def __new__(cls, comp, no, mtm_type, symmetry):
        assert mtm_type in ["MTM", "S2S_MTM"]
        assert symmetry in [0, 1, 2]
        obj = GeneralOperator.__new__(cls, comp, no, mtm_type, symmetry)
        obj._no = no
        obj._mtm_type = mtm_type
        obj._symmetry = Symmetry(symmetry)
        return obj

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
        return "X_({}, {})".format(self.comp, self.no)

    def _print_contents_latex(self, printer):
        return "X_{{{}, {}}}".format(self.comp, self.no)


class Moment(Symbol):
    def __new__(cls, comp, from_state, to_state, op_type):
        assert isinstance(comp, str)
        assert isinstance(from_state, Symbol)
        assert isinstance(to_state, Symbol)
        operator = get_operator_by_name(op_type)
        name = "{}_{}^{}".format(
            operator.symbol, comp, str(from_state) + str(to_state)
        )
        obj = Symbol.__new__(cls, name)
        obj._operator = operator
        obj._comp = comp
        obj._from_state = from_state
        obj._to_state = to_state
    
        if len(obj._comp) != obj._operator.dim:
            raise ValueError(
                f"The operator is {obj._operator.dim}-dimensional, but {len(obj._comp)} "
                "components were specified."
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
        return self._operator.name

    @property
    def symmetry(self):
        return self._operator.symmetry

    @property
    def dim(self):
        return self._operator.dim

    def revert(self):
        if self.symmetry == Symmetry.NOSYMMETRY:
            from_state = self.from_state
            to_state = self.to_state
            sign = 1.0
        else:
            from_state = self.to_state
            to_state = self.from_state
            if self.symmetry == Symmetry.HERMITIAN:
                sign = 1.0
            else:
                assert self.symmetry == Symmetry.ANTIHERMITIAN
                sign = -1.0
        return sign * Moment(self.comp, from_state, to_state, self.op_type)


class TransitionFrequency(Symbol):
    def __new__(self, state, **assumptions):
        assert isinstance(state, Symbol)
        name = "w_{}".format(str(state))
        obj = Symbol.__new__(self, name, **assumptions)
        obj._state = state
        return obj

    @property
    def state(self):
        return self._state
