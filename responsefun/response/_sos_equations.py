#  Copyright (C) 2019 by Maximilian Scheurer
#
#  This file is part of responsefun.
#
#  responsefun is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  responsefun is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with responsefun. If not, see <http:www.gnu.org/licenses/>.
#

from sympy.physics.quantum import operator as qmoperator
from sympy.physics.quantum.state import Bra, Ket
from sympy import Symbol, Mul


def _first_order_sos(op1, op2, n, freq):
    w = Symbol("w_{}".format(str(n)), real=True)
    O = Symbol("0".format(str(n)), real=True)
    if freq == w:
        raise ValueError("Frequencies cannot have identical labels.")
    return (
        - Bra(O) * op1 * Ket(n) * Bra(n) * op2 * Ket(O) / (w - freq)
        - Bra(O) * op2 * Ket(n) * Bra(n) * op1 * Ket(O) / (w + freq)
    )  # eq. 5.308 (Norman book)


# TODO: naming?
def _second_order_sos(op1, op2, op3, n, m, freq1, freq2):
    wn = Symbol("w_{}".format(str(n)), real=True)
    wm = Symbol("w_{}".format(str(m)), real=True)
    wsig = Symbol("w_sigma", real=True)
    O = Symbol("0", real=True)

    op1_shift = shift_operator(op1)
    op2_shift = shift_operator(op2)
    op3_shift = shift_operator(op3)
    if freq1 in [wn, wm] or freq2 in [wn, wm]:
        raise ValueError("Frequencies cannot have identical labels.")

    prelim_expression = (
        Bra(O) * op1 * Ket(m) * Bra(m) * op2_shift * Ket(n) * Bra(n) * op3 * Ket(O) / ((wm - wsig) * (wn - freq2))
        + Bra(O) * op2 * Ket(m) * Bra(m) * op1_shift * Ket(n) * Bra(n) * op3 * Ket(O) / ((wm + freq1) * (wn - freq2))
        + Bra(O) * op2 * Ket(m) * Bra(m) * op3_shift * Ket(n) * Bra(n) * op1 * Ket(O) / ((wm + freq1) * (wn + wsig))
    )  # eq. 5.309 (Norman book)
    perms = [(op2, op3), (op2_shift, op3_shift), (freq1, freq2)]
    perm_dict = {}
    for p in perms:
        perm_dict[p[0]] = p[1]
        perm_dict[p[1]] = p[0]
    return prelim_expression + prelim_expression.subs(perm_dict, simultaneous=True)


def shift_operator(operator):
    if isinstance(operator, qmoperator.HermitianOperator):
        return qmoperator.HermitianOperator(r"\bar{{ {} }}".format(str(operator)))
    elif isinstance(operator, Mul) and len(operator.args) == 2:
        assert isinstance(operator.args[1], qmoperator.HermitianOperator)
        return operator.args[0] * qmoperator.HermitianOperator(r"\bar{{ {} }}".format(str(operator.args[1])))
    else:
        raise NotImplementedError("Cannot build shifted operator.")
