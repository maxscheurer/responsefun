from sympy.physics.quantum.state import Bra, Ket, StateBase
from sympy.physics.quantum.operator import HermitianOperator
import sympy.physics.quantum.operator as qmoperator
from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex
from itertools import permutations
from examples.isr_conversion import TransitionMoment

O, n, k, gamma = symbols("0, n, k, \gamma", real=True)
w_n = Symbol("w_{}".format(str(n)), real=True)
w_k = Symbol("w_{}".format(str(k)), real=True)
w_o = Symbol("w_{\sigma}", real=True)
w_1 = Symbol("w_{1}", real=True)
w_2 = Symbol("w_{2}", real=True)

op_a = qmoperator.HermitianOperator(r"\mu_{\alpha}")
op_b = qmoperator.HermitianOperator(r"\mu_{\beta}")
op_c = qmoperator.HermitianOperator(r"\mu_{\gamma}")

def build_sos_beta():
    beta_real = TransitionMoment(O, op_a, n) * TransitionMoment(n, op_b, k) * TransitionMoment(k, op_c, O) / ((w_n - w_o) * (w_k - w_2))
    perms = list(permutations([(op_a, -w_o), (op_b, w_1), (op_c, w_2)]))
    term = beta_real
    for i, p in enumerate(perms):
        if i == 0:
            continue
        else:
            subs_list = []
            for j, pp in enumerate(p):
                subs_list.append((perms[0][j][0], p[j][0]))
                subs_list.append((perms[0][j][1], p[j][1]))
            new_term = term.subs(subs_list, simultaneous=True)
            beta_real += new_term
    return beta_real