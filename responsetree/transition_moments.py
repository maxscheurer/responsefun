from sympy.physics.quantum.state import Bra, Ket, StateBase
from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex, simplify, fraction
from responsetree.response_operators import DipoleOperator

# TODO: extract k-mer from multiplication...
def extract_bra_op_ket(expr):
    assert type(expr) == Mul
    bok = [Bra, DipoleOperator, Ket]
    expr_types = [type(term) for term in expr.args]
    ret = [list(expr.args[i:i+3]) for i, k in enumerate(expr_types)
           if expr_types[i:i+3] == bok]
    return ret


class TransitionMoment:
    def __init__(self, from_state, operator, to_state):
        self.expr = Bra(from_state) * operator * Ket(to_state)

    def __rmul__(self, other):
        return other * self.expr

    def __mul__(self, other):
        return self.expr * other

    def __repr__(self):
        return str(self.expr)

