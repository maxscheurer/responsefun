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

from responsefun.response.response_functions import ResponseFunction

polarizability = ResponseFunction(r"<<\mu_\alpha;-\mu_\beta>>", [r"\omega_"])
polarizability.sum_over_states.set_frequencies([0])

f = open("bla.tex", "w")
f.write(polarizability.sum_over_states.latex)
f.close()