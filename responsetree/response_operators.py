import sympy.physics.quantum.operator as qmoperator


class ResponseOperator(qmoperator.Operator):
    def __init__(self, comp):
        """
        Base class for (state to state) modified transition moments and response vectors
        :param comp: Cartesian component
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

    def _print_contents(self, printer):
        return "X_{{{}, {}}}".format(self._comp, str(self._no))

    def _print_contents_latex(self, printer):
        return "X_{{{}, {}}}".format(self._comp, str(self._no))


class Matrix(qmoperator.Operator):
    pass


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
