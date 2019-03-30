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

from unittest import TestCase

import sympy.physics.quantum.operator as qmoperator
from sympy import symbols, Symbol

from responsefun.response.sum_over_states import SumOverStatesExpression


class TestSumOverStatesExpression(TestCase):

    def test_create_sos_first_order(self):
        test_op_a = qmoperator.HermitianOperator(r"\mu_{\alpha}")
        test_op_b = qmoperator.HermitianOperator(r"\mu_{\beta}")
        w = Symbol("w", real=True)
        sos = SumOverStatesExpression(summation_indices=["n"], operators=[test_op_a, test_op_b],
                                      frequencies=[w])
        assert sos.number_of_terms == 2

    def test_create_sos_second_order(self):
        test_op_a = qmoperator.HermitianOperator(r"\mu_{\alpha}")
        test_op_b = -qmoperator.HermitianOperator(r"\mu_{\beta}")
        test_op_c = -qmoperator.HermitianOperator(r"\mu_{\gamma}")
        w1, w2 = symbols("w_1 w_2", real=True)
        sos = SumOverStatesExpression(summation_indices=["n", "m"], operators=[test_op_a, test_op_b, test_op_c],
                                      frequencies=[w1, w2])
        assert sos.number_of_terms == 6
