from sympy import Symbol
import sympy.physics.quantum.operator as qmoperator


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
    def __init__(self, comp, op_type="electric"):
        super().__init__(comp)
        assert op_type in ["electric", "magnetic"] or op_type in [Symbol("electric"), Symbol("magnetic")]
        self._op_type = op_type
        
    @property
    def op_type(self):
        return self._op_type

    def _print_contents(self, printer):
        if self._op_type == "electric":
            return "F_{{{}}}".format(self._comp)
        else:
            return "Fm_{{{}}}".format(self._comp)

    def _print_contents_latex(self, printer):
        if self._op_type == "electric":
            return "F_{{{}}}".format(self._comp)
        else:
            "Fm_{{{}}}".format(self._comp)


class S2S_MTM(ResponseOperator):
    def __init__(self, comp, op_type="electric"):
        super().__init__(comp)
        assert op_type in ["electric", "magnetic"] or op_type in [Symbol("electric"), Symbol("magnetic")]
        self._op_type = op_type

    @property
    def op_type(self):
        return self._op_type

    def _print_contents(self, printer):
        if self._op_type == "electric":
            return "B_{{{}}}".format(self._comp)
        else:
            return "Bm_{{{}}}".format(self._comp)

    def _print_contents_latex(self, printer):
        if self._op_type == "electric":
            return "B_{{{}}}".format(self._comp)
        else:
            return "Bm_{{{}}}".format(self._comp)


class ResponseVector(ResponseOperator):
    def __init__(self, comp, no=None):
        super().__init__(comp)
        self._no = no

    @property
    def no(self):
        return self._no

    def _print_contents(self, printer):
        return "X_{{{}, {}}}".format(self._comp, self._no)

    def _print_contents_latex(self, printer):
        return "X_{{{}, {}}}".format(self._comp, self._no)


class DipoleOperator(qmoperator.HermitianOperator):
    def __init__(self, comp, op_type="electric"):
        assert op_type in ["electric", "magnetic"] or op_type in [Symbol("electric"), Symbol("magnetic")]
        self._comp = comp
        self._op_type = op_type

    @property
    def comp(self):
        return self._comp

    @property
    def op_type(self):
        return self._op_type

    def _print_contents(self, printer):
        if self._op_type == "electric":
            return r"\mu_{{{}}}".format(self._comp)
        else:
            return r"m_{{{}}}".format(self._comp)

    def _print_contents_latex(self, printer):
        if self._op_type == "electric":
            return r"\mu_{{{}}}".format(self._comp)
        else:
            return r"m_{{{}}}".format(self._comp)


class DipoleMoment(Symbol):
    def __new__(self, comp, from_state, to_state, op_type="electric", **assumptions):
        assert type(comp) == str
        assert type(from_state) == str
        assert type(to_state) == str
        assert op_type in ["electric", "magnetic"]
        if op_type == "electric":
            name = r"\mu_{{{}}}^{{{}}}".format(comp, from_state+to_state)
        else:
            name = r"m_{{{}}}^{{{}}}".format(comp, from_state+to_state)
        obj = Symbol.__new__(self, name, **assumptions)
        obj._comp = comp
        obj._from_state = from_state
        obj._to_state = to_state
        obj._op_type = op_type
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
