#  Copyright (C) 2023 by the responsefun authors
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

from anytree import NodeMixin, RenderTree
from sympy import Add, Mul, Pow, adjoint
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import Bra, Ket

from responsefun.AdccProperties import available_operators
from responsefun.ResponseOperator import MTM, S2S_MTM, ResponseVector
from responsefun.symbols_and_labels import M, gamma


class IsrTreeNode(NodeMixin):
    def __init__(self, expr, parent=None, children=None):
        """
        Parameters
        ----------
        expr: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
            SymPy expression the node represents.

        parent: <class 'anytree.node.nodemixin.NodeMixin'>
            Parent node.

        children: iterable of <class 'anytree.node.nodemixin.NodeMixin'>
            List of child nodes.
        """
        super().__init__()
        self.expr = expr
        self.parent = parent
        if children:
            self.children = children


class ResponseNode(NodeMixin):
    def __init__(self, expr, tinv, rhs, parent=None):
        """
        Parameters
        ----------
        expr: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
            SymPy expression the node represents.

        tinv: <class 'sympy.core.mul.Mul'>
            Term containing the inverse (shifted) ADC matrix.

        rhs: <class 'responsefun.ResponseOperator.MTM'> or sympy.physics.quantum.dagger.Dagger
            or <class 'sympy.core.mul.Mul'>
            Rhs of the response equation to be solved.

        parent: <class 'anytree.node.nodemixin.NodeMixin'>
            Parent node.
        """
        super().__init__()
        assert isinstance(expr, Mul)
        self.expr = expr
        self.tinv = tinv
        self.rhs = rhs  # rhs of response equation
        self.w = tinv.subs([(M, 0), (gamma, 0)])
        self.gamma = tinv.subs([(M, 0), (self.w, 0)])
        self.parent = parent


def acceptable_rhs_lhs(term):
    if isinstance(term, adjoint):
        op_expr = term.args[0]
    else:
        op_expr = term
    return isinstance(op_expr, MTM) or isinstance(op_expr, ResponseVector)


def acceptable_two_rhss_lhss(term1, term2):
    if isinstance(term1, S2S_MTM):
        op_term2 = term2
        if isinstance(term2, adjoint):
            op_term2 = term2.args[0]
        if isinstance(op_term2, Bra) or isinstance(op_term2, Ket):  # <f|B or B|f>
            return True
        elif isinstance(op_term2, ResponseVector):  # Dagger(X) * B or B * X
            return True
        else:
            return False
    else:
        return False


def build_branches(node, matrix):
    """Find response equations to be solved by building up a tree structure."""
    if isinstance(node.expr, Add):
        node.children = [IsrTreeNode(term) for term in node.expr.args]
        for child in node.children:
            build_branches(child, matrix)
    elif isinstance(node.expr, Mul):
        children = []
        for i, term in enumerate(node.expr.args):
            if isinstance(term, Pow) and (matrix in term.args[0].args or term.args[0] == matrix):
                tinv = term.args[0]
                lhs = node.expr.args[i - 1]
                rhs = node.expr.args[i + 1]
                if term.args[1] != -1:
                    if acceptable_rhs_lhs(rhs):
                        children.append(ResponseNode(tinv**-1 * rhs, tinv, rhs))
                    elif acceptable_two_rhss_lhss(rhs, node.expr.args[i + 2]):
                        children.append(
                            ResponseNode(
                                tinv**-1 * rhs * node.expr.args[i + 2],
                                tinv,
                                rhs * node.expr.args[i + 2],
                            )
                        )
                    else:
                        print("No invertable term found.")
                    if acceptable_rhs_lhs(lhs):
                        children.append(ResponseNode(lhs * tinv**-1, tinv, lhs))
                    elif acceptable_two_rhss_lhss(lhs, node.expr.args[i - 2]):
                        children.append(
                            ResponseNode(
                                node.expr.args[i - 2] * lhs * tinv**-1,
                                tinv,
                                node.expr.args[i - 2] * lhs,
                            )
                        )
                    else:
                        print("No invertable term found.")
                else:
                    if acceptable_rhs_lhs(rhs):
                        children.append(ResponseNode(tinv**-1 * rhs, tinv, rhs))
                    elif acceptable_rhs_lhs(lhs):
                        children.append(ResponseNode(lhs * tinv**-1, tinv, lhs))
                    elif acceptable_two_rhss_lhss(rhs, node.expr.args[i + 2]):
                        children.append(
                            ResponseNode(
                                tinv**-1 * rhs * node.expr.args[i + 2],
                                tinv,
                                rhs * node.expr.args[i + 2],
                            )
                        )
                    elif acceptable_two_rhss_lhss(lhs, node.expr.args[i - 2]):
                        children.append(
                            ResponseNode(
                                node.expr.args[i - 2] * lhs * tinv**-1,
                                tinv,
                                node.expr.args[i - 2] * lhs,
                            )
                        )
                    else:
                        print("No invertable term found.")
        node.children = children
    else:
        raise TypeError("ADC/ISR expression must be either of type Mul or Add.")


def traverse_branches(node, old_expr, new_expr):
    """Traverse the branch and replace the leaf expression in each node."""
    oe = node.expr  # new "old expression"
    ne = node.expr.subs(old_expr, new_expr)  # new "new expression"
    node.expr = ne
    # keep traversing the branch if the root has not yet been reached
    if not node.is_root:
        traverse_branches(node.parent, oe, ne)


def show_tree(root):
    for pre, _, node in RenderTree(root):
        treestr = "%s%s" % (pre, node.expr)
        print(treestr.ljust(8))


def build_tree(isr_expression, matrix=Operator("M"), rvecs_list=None, no=1):
    """Build a tree structure to define response vectors for evaluating
    the ADC/ISR formulation of a molecular property.

    Parameters
    ----------
    isr_expression: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
        SymPy expression of the ADC/ISR formulation.

    matrix: <class 'sympy.physics.quantum.operator.Operator'>, optional
        The matrix contained in the SymPy expression.

    rvecs_list: list, optional
        only for recursive function call

    no: int, optional
        only for recursive function call

    Returns
    ----------
    list of tuples
        For each tuple: The first entry is the root expression, i.e., a SymPy expression that contains instances
        of <class 'responsefun.ResponseOperator.ResponseVector'>;
        the second entry is a dictionary with tuples as keys specifying the response vectors.
    """
    if rvecs_list is None:
        rvecs_list = []
    root = IsrTreeNode(isr_expression)
    build_branches(root, matrix)
    show_tree(root)
    rvecs = {}

    for leaf in root.leaves:
        if not isinstance(leaf, ResponseNode):
            continue
        # if the leaf node is an instance of the ResponseNode class, a tuple will be defined
        # that uniquely describes the resulting response vector
        old_expr = leaf.expr
        oper_rhs = leaf.rhs
        with_dagger = None
        if isinstance(leaf.rhs, adjoint):
            oper_rhs = leaf.rhs.args[0]

        if isinstance(oper_rhs, Mul):
            if isinstance(oper_rhs.args[0], S2S_MTM):
                with_dagger = False
                if isinstance(oper_rhs.args[1], ResponseVector):
                    key = (
                        oper_rhs.args[0].__class__.__name__,
                        oper_rhs.args[0].op_type,
                        leaf.w,
                        leaf.gamma,
                        oper_rhs.args[1].__class__.__name__,
                        oper_rhs.args[1].no,
                    )
                    comp = oper_rhs.args[0].comp + oper_rhs.args[1].comp
                else:
                    key = (
                        oper_rhs.args[0].__class__.__name__,
                        oper_rhs.args[0].op_type,
                        leaf.w,
                        leaf.gamma,
                        oper_rhs.args[1].label[0],
                        None,
                    )
                    comp = oper_rhs.args[0].comp

            elif isinstance(oper_rhs.args[1], S2S_MTM):
                with_dagger = True
                oper_rhs2 = oper_rhs.args[0]
                if isinstance(oper_rhs2, adjoint):
                    oper_rhs2 = oper_rhs.args[0].args[0]
                if isinstance(oper_rhs2, ResponseVector):
                    key = (
                        oper_rhs.args[1].__class__.__name__,
                        oper_rhs.args[1].op_type,
                        leaf.w,
                        leaf.gamma,
                        oper_rhs2.__class__.__name__,
                        oper_rhs2.no,
                    )
                    comp = oper_rhs.args[1].comp + oper_rhs2.comp
                else:
                    key = (
                        oper_rhs.args[1].__class__.__name__,
                        oper_rhs.args[1].op_type,
                        leaf.w,
                        leaf.gamma,
                        oper_rhs2.label[0],
                        None,
                    )
                    comp = oper_rhs.args[1].comp

            else:
                raise ValueError()

        elif isinstance(oper_rhs, ResponseVector):
            key = (oper_rhs.__class__.__name__, None, leaf.w, leaf.gamma, None, oper_rhs.no)
            comp = oper_rhs.comp

        else:
            key = (oper_rhs.__class__.__name__, oper_rhs.op_type, leaf.w, leaf.gamma, None, None)
            comp = oper_rhs.comp

        # if the created tuple is not already among the keys of the rvecs dictionary, a new entry will be made
        if key not in rvecs:
            rvecs[key] = no
            no += 1

        if with_dagger is None:
            if oper_rhs == leaf.rhs:
                with_dagger = False
            else:
                with_dagger = True
        mtm_type = key[0]
        op_type = key[1]
        symmetry = available_operators[op_type][1]
        if not with_dagger:
            leaf.expr = ResponseVector(comp, rvecs[key], mtm_type, symmetry)
        else:
            leaf.expr = adjoint(ResponseVector(comp, rvecs[key], mtm_type, symmetry))
        traverse_branches(leaf.parent, old_expr, leaf.expr)

    if rvecs:
        rvecs_list.append((root.expr, rvecs))
        build_tree(root.expr, matrix, rvecs_list, no)
    return rvecs_list
