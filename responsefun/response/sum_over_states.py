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
from sympy import Symbol, latex, Mul, Add

from ._sos_equations import _first_order_sos, _second_order_sos


class SumOverStatesExpression:

    __expression_templates = {
        1: _first_order_sos,
        2: _second_order_sos,
    }

    def __init__(self, summation_indices, operators, frequencies):
        """
        Class representing sum-over-states (SOS) expressions

        :param summation_indices: list of strings for indices of the excited states to sum over (same len as order)
        :param operators: list of operators (sympy HermitianOperator)
        """
        order = len(summation_indices)
        if order != len(frequencies):
            raise ValueError("Need to have as many summation indices as frequencies.")
        if order > 2:
            raise NotImplementedError("SOS only implemented through second order.")

        for op in operators:
            if not isinstance(op, qmoperator.HermitianOperator):
                if isinstance(op, Mul):
                    have_valid_operator = False
                    for comps in op.args:
                        if isinstance(comps, qmoperator.HermitianOperator):
                            have_valid_operator = True
                        elif isinstance(comps, Add):
                            raise TypeError("Addition not supported in operator list.")
                    if not have_valid_operator:
                        raise TypeError("Operators must be of type sympy.physics.quantum.HermitianOperator or Mul"
                                        "with HermitianOperator")
                else:
                    raise TypeError("Operators must be of type sympy.physics.quantum.HermitianOperator or Mul"
                                    "with HermitianOperator")

        indices = []
        for idx in summation_indices:
            indices.append(Symbol(idx))

        sos_template = self.__expression_templates[order]

        self.expression = sos_template(*operators, *indices, *frequencies)

    @property
    def number_of_terms(self):
        return len(self.expression.args)

    @property
    def latex(self):
        return latex(self.expression)
