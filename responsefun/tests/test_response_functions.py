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
import pytest

from sympy.physics.quantum import operator as qmoperator
from sympy import Symbol

from responsefun.response.response_functions import ResponseFunction, sympify_operator


class TestResponseFunction(TestCase):

    def test_create_response_functions(self):
        operator_string = r"\mu_\alpha"
        pert1_string = r"-\mu_\beta"
        pert2_string = r"-\mu_\gamma"
        freq1 = "w_1"
        freq2 = "w_2"
        freqs = [freq1, freq2]
        rspfun = ResponseFunction("<<{};{},{}>>".format(operator_string,
                                                        pert1_string, pert2_string),
                                  freqs)
        # TODO: testing only
        # w = Symbol("\omega")
        # rspfun.sum_over_states.set_frequencies([w, w])
        # f = open("bla.tex", "w")
        # f.write(rspfun.sum_over_states.latex)
        # f.close()

        operator_ref = sympify_operator(operator_string)
        perturbations_ref = [sympify_operator(pert1_string),
                             sympify_operator(pert2_string)]

        assert len(rspfun.perturbation_frequencies)

        for test, ref in zip(rspfun.perturbation_operators, perturbations_ref):
            assert test == ref

        for test, ref in zip(rspfun.perturbation_frequencies, freqs):
            assert test == Symbol(ref, real=True)

        assert operator_ref == rspfun.prop_operator

        with pytest.raises(Exception):
            ResponseFunction("<<mu_alpha;>>", [])

        with pytest.raises(Exception):
            ResponseFunction("<<mu_alpha;a,b,c>>", [])

    def test_sympify_operator(self):
        str1 = r"-\mu_\gamma"
        op = sympify_operator(str1)

        op_ref = -qmoperator.HermitianOperator(r"\mu_\gamma")
        assert op == op_ref

        str1 = r"\mu_\gamma"
        op = sympify_operator(str1)

        op_ref = qmoperator.HermitianOperator(r"\mu_\gamma")
        assert op == op_ref




