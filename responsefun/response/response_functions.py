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
from typing import List

from sympy.physics.quantum import operator as qmoperator
from sympy import Symbol


class ResponseFunction:
    """
    Class representing a general response function.
    It can be easily constructed from a string.
    """
    def __init__(self, string=None, frequencies=None):
        """
        Constructor from string
        :param string: string representation of the response function

        Example:
            ResponseFunction("<<mu_alpha;mu_beta>>", ["w"])
        """
        self._order, self.prop_operator, self.perturbation_operators = \
            build_response_function_operators_from_string(string)
        if self._order > 2:
            raise NotImplementedError("Response functions are only implemented through second order.")

        self.outgoing_frequency, self.perturbation_frequencies = build_frequencies_from_list(frequencies)
        if len(self.perturbation_frequencies) > self._order:
            raise ValueError("Invalid number of frequencies specified. Need {} frequencies, not {}.".format(
                self._order, len(self.perturbation_frequencies)
            ))

        # TODO: build SOS expression for response function


def build_response_function_operators_from_string(string: str) -> tuple:
    """
    Build the operators of a response function from string
    :param string: String representation of the response function
    :return: tuple (order, property operator, perturbation operators)
    """
    components = string.strip("<<").strip(">>").split(";")
    assert len(components) == 2
    prop_operator_string = components[0]  # the operator \Omega whose expectation value is of interest
    perturbation_string_list = [x for x in components[1].split(",") if x != '']
    prop_operator = qmoperator.HermitianOperator(prop_operator_string)
    perturbation_operators = []
    if len(perturbation_string_list) == 0:
        raise ValueError("Invalid response function specified. No perturbation found.")
    for pert_string in perturbation_string_list:
        perturbation_operators.append(qmoperator.HermitianOperator(pert_string))
    return len(perturbation_string_list), prop_operator, perturbation_operators


def build_frequencies_from_list(frequencies: List[str]) -> tuple:
    """
    Builds the outgoing frequency from a string list labeling incoming (perturbation)
    frequencies.
    Note that w_outgoing = sum of all incoming frequencies.
    In the common notation, one finds, e.g. \beta(-w_sigma; w_1, w_2), i.e.,
    the outgoing frequency is saved here with the opposite (positive) sign

    :param frequencies: list of strings labeling frequencies
    :return: tuple (outgoing_frequency, perturbation_frequences) as sympy.Symbol
    """

    perturbation_frequencies: List[Symbol] = []
    outgoing_frequency = 0
    if not isinstance(frequencies, list):
        raise TypeError("Invalid type specified. frequencies must be a list.")

    for w in frequencies:
        incoming = Symbol(w, real=True)
        perturbation_frequencies.append(incoming)
        outgoing_frequency += incoming
    return outgoing_frequency, perturbation_frequencies
