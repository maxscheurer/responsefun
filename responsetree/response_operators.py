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
    def __init__(self, comp):
        super().__init__(comp)

    def _print_contents(self, printer):
        return "F_{{{}}}".format(self._comp)

    def _print_contents_latex(self, printer):
        return "F_{{{}}}".format(self._comp)


class S2S_MTM(ResponseOperator):
    def __init__(self, comp):
        super().__init__(comp)

    def _print_contents(self, printer):
        return "B_{{{}}}".format(self._comp)

    def _print_contents_latex(self, printer):
        return "B_{{{}}}".format(self._comp)


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
    def __init__(self, comp):
        self._comp = comp

    @property
    def comp(self):
        return self._comp

    def _print_contents(self, printer):
        return "\mu_{{{}}}".format(self._comp)

    def _print_contents_latex(self, printer):
        return "\mu_{{{}}}".format(self._comp)


class DipoleMoment(Symbol):
    def __new__(self, comp, from_state, to_state, **assumptions):
        assert type(comp) == str
        assert type(from_state) == str
        assert type(to_state) == str
        name = r"\mu_{{{}}}^{{{}}}".format(comp, from_state+to_state)
        obj = Symbol.__new__(self, name, **assumptions)
        obj._comp = comp
        obj._from_state = from_state
        obj._to_state = to_state
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
