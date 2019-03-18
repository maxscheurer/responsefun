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

from responsefun.response import response_functions


class TestResponseFunction(TestCase):

    def test_create_response_functions(self):
        operator_string = "mu_alpha"
        pert1_string = "mu_beta"
        pert2_string = "mu_gamma"
        freq1 = "w_1"
        freq2 = "w_2"
        freqs = [freq1, freq2]
        rspfun = response_functions.ResponseFunction("<<{};{},{}>>".format(operator_string,
                                                                           pert1_string, pert2_string),
                                                     freqs)

        operator_ref = qmoperator.HermitianOperator(operator_string)
        perturbations_ref = [qmoperator.HermitianOperator(pert1_string), qmoperator.HermitianOperator(pert2_string)]

        assert len(rspfun.perturbation_frequencies)

        for test, ref in zip(rspfun.perturbation_operators, perturbations_ref):
            assert test == ref

        for test, ref in zip(rspfun.perturbation_frequencies, freqs):
            assert test == Symbol(ref, real=True)

        assert operator_ref == rspfun.prop_operator

        with pytest.raises(Exception):
            response_functions.ResponseFunction("<<mu_alpha;>>", [])

        with pytest.raises(Exception):
            response_functions.ResponseFunction("<<mu_alpha;a,b,c>>", [])



