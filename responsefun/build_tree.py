from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex, Abs, UnevaluatedExpr
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import Bra, Ket, StateBase
from anytree import NodeMixin, RenderTree

from responsefun.symbols_and_labels import *
from responsefun.response_operators import MTM, S2S_MTM, ResponseVector
from responsefun.adcc_properties import available_operators


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

        rhs: <class 'responsetree.response_operators.MTM'> or sympy.physics.quantum.dagger.Dagger or <class 'sympy.core.mul.Mul'>
            Rhs of the response equation to be solved.

        parent: <class 'anytree.node.nodemixin.NodeMixin'>
            Parent node.
        """
        super().__init__()
        assert isinstance(expr, Mul)
        self.expr = expr
        self.tinv = tinv
        self.rhs = rhs #rhs of response equation
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
        if isinstance(op_term2, Bra) or isinstance(op_term2, Ket): # <f|B or B|f> 
            return True
        elif isinstance(op_term2, ResponseVector): # Dagger(X) * B or B * X
            return True
        else:
            return False
    else:
        return False


def build_branches(node, matrix):
    """Find response equations to be solved by building up a tree structure.
    """
    if isinstance(node.expr, Add):
        node.children = [IsrTreeNode(term) for term in node.expr.args]
        for child in node.children:
            build_branches(child, matrix)
    elif isinstance(node.expr, Mul):
        children = []
        for i, term in enumerate(node.expr.args):
            if isinstance(term, Pow) and (matrix in term.args[0].args or term.args[0]==matrix):
                tinv = term.args[0]
                lhs = node.expr.args[i-1]
                rhs = node.expr.args[i+1]
                if term.args[1] != -1:
                    if acceptable_rhs_lhs(rhs):
                        children.append(ResponseNode(tinv**-1 * rhs, tinv, rhs))
                    elif acceptable_two_rhss_lhss(rhs, node.expr.args[i+2]):
                        children.append(ResponseNode(tinv**-1 * rhs * node.expr.args[i+2], tinv, rhs * node.expr.args[i+2]))
                    else:
                        print("No invertable term found.")
                    if acceptable_rhs_lhs(lhs):
                        children.append(ResponseNode(lhs * tinv**-1, tinv, lhs))
                    elif acceptable_two_rhss_lhss(lhs, node.expr.args[i-2]):
                        children.append(ResponseNode(node.expr.args[i-2] * lhs * tinv**-1, tinv, node.expr.args[i-2] * lhs))
                    else:
                        print("No invertable term found.")
                else:
                    if acceptable_rhs_lhs(rhs):
                        children.append(ResponseNode(tinv**-1 * rhs, tinv, rhs))
                    elif acceptable_rhs_lhs(lhs):
                        children.append(ResponseNode(lhs * tinv**-1, tinv, lhs))
                    elif acceptable_two_rhss_lhss(rhs, node.expr.args[i+2]):
                        children.append(ResponseNode(tinv**-1 * rhs * node.expr.args[i+2], tinv, rhs * node.expr.args[i+2]))
                    elif acceptable_two_rhss_lhss(lhs, node.expr.args[i-2]):
                        children.append(ResponseNode(node.expr.args[i-2] * lhs * tinv**-1, tinv, node.expr.args[i-2] * lhs))
                    else:
                        print("No invertable term found.")
        node.children = children
    else:
        raise TypeError("ADC/ISR expression must be either of type Mul or Add.")
                    

def traverse_branches(node, old_expr, new_expr):
    """Traverse the branch and replace the leaf expression in each node.
    """
    oe = node.expr # new "old expression"
    ne = node.expr.subs(old_expr, new_expr) # new "new expression"
    node.expr = ne
    # keep traversing the branch if the root has not yet been reached
    if not node.is_root:
        traverse_branches(node.parent, oe, ne)


def show_tree(root):
    for pre, _, node in RenderTree(root):
        treestr = u"%s%s" % (pre, node.expr)
        print(treestr.ljust(8))


def build_tree(isr_expression, matrix=Operator("M"), rvecs_list=None, no=1):
    """Build a tree structure to define response vectors for evaluating the ADC/ISR formulation of a molecular property.
    
    Parameters
    ----------
    isr_expression: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
        SymPy expression of the ADC/ISR formulation.

    matrix: <class 'sympy.physics.quantum.operator.Operator'>, optional
        The matrix contained in the SymPy expression.

    rvecs_list: list, optional
        bla

    no: int, optional
        bla

    Returns
    ----------
    list of tuples
        For each tuple: The first entry is the root expression, i.e., a SymPy expression that contains instances
        of <class 'responsetree.response_operators.ResponseVector'>;
        the second entry is a dictionary with tuples as keys specifying the response vectors.
    """
    if rvecs_list is None:
        rvecs_list = []
    root = IsrTreeNode(isr_expression)
    build_branches(root, matrix)
    rvecs = {}

    for leaf in root.leaves:
        if not isinstance(leaf, ResponseNode):
            continue
        # if the leaf node is an instance of the ResponseNode class, a tuple will be defined that uniquely describes the resulting response vector
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
                            oper_rhs.args[0].__class__.__name__, oper_rhs.args[0].op_type,
                            leaf.w, leaf.gamma, oper_rhs.args[1].__class__.__name__, oper_rhs.args[1].no
                    )
                    comp = oper_rhs.args[0].comp + oper_rhs.args[1].comp
                else:
                    key = (
                            oper_rhs.args[0].__class__.__name__, oper_rhs.args[0].op_type,
                            leaf.w, leaf.gamma, oper_rhs.args[1].label[0], None
                    )
                    comp = oper_rhs.args[0].comp

            elif isinstance(oper_rhs.args[1], S2S_MTM):
                with_dagger = True
                oper_rhs2 = oper_rhs.args[0]
                if isinstance(oper_rhs2, adjoint):
                    oper_rhs2 = oper_rhs.args[0].args[0]
                if isinstance(oper_rhs2, ResponseVector):
                    key = (
                            oper_rhs.args[1].__class__.__name__, oper_rhs.args[1].op_type,
                            leaf.w, leaf.gamma, oper_rhs2.__class__.__name__, oper_rhs2.no
                    )
                    comp = oper_rhs.args[1].comp + oper_rhs2.comp
                else:
                    key = (
                            oper_rhs.args[1].__class__.__name__, oper_rhs.args[1].op_type,
                            leaf.w, leaf.gamma, oper_rhs2.label[0], None
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
        show_tree(root)
        rvecs_list.append((root.expr, rvecs))
        build_tree(root.expr, matrix, rvecs_list, no)
    return rvecs_list


if __name__ == "__main__":
    alpha_like = adjoint(F_A) * (M - w - 1j*gamma)**-1 * F_B + adjoint(F_B) * (M + w + 1j*gamma)**-1 * F_A
    beta_like = adjoint(F_A) * (M - w)**-1 * B_B * (M + w)**-1 * F_C
    beta_real = (
        adjoint(F_A) * (M - w_o)**-1 * B_B * (M - w_2)**-1 * F_C
        + adjoint(F_A) * (M - w_o)**-1 * B_C * (M - w_1)**-1 * F_B
        + adjoint(F_C) * (M + w_2)**-1 * B_B * (M + w_o)**-1 * F_A
        + adjoint(F_B) * (M + w_1)**-1 * B_C * (M + w_o)**-1 * F_A
        + adjoint(F_B) * (M + w_1)**-1 * B_A * (M - w_2)**-1 * F_C
        + adjoint(F_C) * (M + w_2)**-1 * B_A * (M - w_1)**-1 * F_B
    )
    gamma_like = adjoint(F_A) * (M - w)**-1 * B_B * (M + w)**-1 * B_D * (M + 2*w)**-1 * F_C + adjoint(F_D) * (M - w)**-1 * B_A * (M + w)**-1 * B_C * (M + 2*w)**-1 * F_B
    gamma_extra = adjoint(F_A) * (M - w_o)**-1 * F_B * adjoint(F_C) * (M - w_3)**-1 * F_D / (-w_2 - w_3)
    gamma_extra2 = adjoint(F_A) * (M)**-1 * F_B * adjoint(F_C) * (M)**-1 * (M)**-1 * F_D
    B_E = S2S_MTM("E", "electric")
    F_F = MTM("F", "electric")
    higher_order_like = adjoint(F_A) * (M - w)**-1 * B_B * (M - w)**-1 * B_C * (M - w)**-1 * B_D * (M - w)**-1 * B_E * (M - w)**-1 * F_F
    higher_order_extra_like = adjoint(F_A) * (M - w_o)**-1 * F_B * adjoint(F_C) * (M + w_o)**-1 * (M + w_2)**-1 * (M - w_3)**-1 * (M + w)**-1 * F_D
    #print(build_tree(alpha_like))
    #print(build_tree(beta_like))
    #print(build_tree(beta_real))
    #build_tree(gamma_like)
    #print(build_tree(gamma_extra))
    #print(build_tree(higher_order_like))
    #print(build_tree(gamma_extra2))
    #print(build_tree(higher_order_extra_like))
