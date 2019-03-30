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


def _first_order_sos(opo, op1, n, freq, complex=False):
    """
    First-order sum-over-states expression
    eq. 5.308 (Norman book)

    :param opo: omega operator
    :param op1: perturbation operator
    :param n: state index n
    :param freq: frequency of perturbation
    :param complex: bool to get complex function
    :return: sympy expression of the full SOS
    """
    if complex:
        raise NotImplementedError("Complex SOS expressions are not implemented.")
    wn = Symbol("w_{}".format(str(n)), real=True)
    O = Symbol("0".format(str(n)), real=True)
    if freq == wn:
        raise ValueError("Frequencies cannot have identical labels.")
    return (
        - Bra(O) * opo * Ket(n) * Bra(n) * op1 * Ket(O) / (wn - freq)
        - Bra(O) * op1 * Ket(n) * Bra(n) * opo * Ket(O) / (wn + freq)
    ), freq


# TODO: naming?
def _second_order_sos(opo, op1, op2, n, m, freq1, freq2, complex=False):
    """
    Second-order sum-over-states expression
    eq. 5.309 (Norman book)

    :param opo: omega operator
    :param op1: perturbation operator 1
    :param op2: perturbation operator 2
    :param n: state index n
    :param m: state index m
    :param freq1: frequency of perturbation 1
    :param freq2: frequency of perturbation 2
    :param complex: bool to get complex function
    :return: sympy expression of the full SOS
    """
    if complex:
        raise NotImplementedError("Complex SOS expressions are not implemented.")
    wn = Symbol("w_{}".format(str(n)), real=True)
    wm = Symbol("w_{}".format(str(m)), real=True)
    wsig = Symbol("w_sigma", real=True)
    O = Symbol("0", real=True)

    opo_shift = shift_operator(opo)
    op1_shift = shift_operator(op1)
    op2_shift = shift_operator(op2)
    if freq1 in [wn, wm] or freq2 in [wn, wm]:
        raise ValueError("Frequencies cannot have identical labels.")

    prelim_expression = (
        Bra(O) * opo * Ket(m) * Bra(m) * op1_shift * Ket(n) * Bra(n) * op2 * Ket(O) / ((wm - wsig) * (wn - freq2))
        + Bra(O) * op1 * Ket(m) * Bra(m) * opo_shift * Ket(n) * Bra(n) * op2 * Ket(O) / ((wm + freq1) * (wn - freq2))
        + Bra(O) * op1 * Ket(m) * Bra(m) * op2_shift * Ket(n) * Bra(n) * opo * Ket(O) / ((wm + freq1) * (wn + wsig))
    )
    perms = [(op1, op2), (op1_shift, op2_shift), (freq1, freq2)]
    perm_dict = {}
    for p in perms:
        perm_dict[p[0]] = p[1]
        perm_dict[p[1]] = p[0]
    return prelim_expression + prelim_expression.subs(perm_dict, simultaneous=True), wsig


def shift_operator(operator):
    if isinstance(operator, qmoperator.HermitianOperator):
        return qmoperator.HermitianOperator(r"\bar{{ {} }}".format(str(operator)))
    elif isinstance(operator, Mul) and len(operator.args) == 2:
        assert isinstance(operator.args[1], qmoperator.HermitianOperator)
        return operator.args[0] * qmoperator.HermitianOperator(r"\bar{{ {} }}".format(str(operator.args[1])))
    else:
        raise NotImplementedError("Cannot build shifted operator.")
