from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import Bra, Ket, StateBase
from anytree import NodeMixin, RenderTree

from responsetree.symbols_and_labels import *
from responsetree.response_operators import MTM, S2S_MTM, ResponseVector


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


def acceptable_rhs_lhs_MTM(term):
    if isinstance(term, adjoint):
        op_expr = term.args[0]
    else:
        op_expr = term
    return isinstance(op_expr, MTM)


def acceptable_rhs_lhs_S2S_MTM(term1, term2):
    return isinstance(term1, S2S_MTM) and (isinstance(term2, Bra) or isinstance(term2, Ket))


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
            if isinstance(term, Pow) and term.args[1] == -1 and (matrix in term.args[0].args or term.args[0]==matrix):
                tinv = term.args[0]
                lhs = node.expr.args[i-1]
                rhs = node.expr.args[i+1]
                if acceptable_rhs_lhs_MTM(rhs):
                    children.append(ResponseNode(tinv**-1 * rhs, tinv, rhs))
                elif acceptable_rhs_lhs_MTM(lhs):
                    children.append(ResponseNode(lhs * tinv**-1, tinv, lhs))
                elif acceptable_rhs_lhs_S2S_MTM(rhs, node.expr.args[i+2]):
                    children.append(ResponseNode(tinv**-1 * rhs * node.expr.args[i+2], tinv, rhs * node.expr.args[i+2]))
                elif acceptable_rhs_lhs_S2S_MTM(lhs, node.expr.args[i-2]):
                    children.append(ResponseNode(node.expr.args[i-2] * lhs * tinv**-1, tinv, node.expr.args[i-2] * lhs))
                else:
                    print("No invertable term found.")
        node.children = children
    else:
        raise TypeError("ADC/ISR expression must be either of type Mul or Add.")
                    

def traverse_branches(node, old_expr, new_expr):
    """Traverse the branch and replace the leaf expression in each node.
    """
    oe = node.expr
    ne = node.expr.subs(old_expr, new_expr)
    node.expr = ne
    if not node.is_root:
        traverse_branches(node.parent, oe, ne)


def show_tree(root):
    for pre, _, node in RenderTree(root):
        treestr = u"%s%s" % (pre, node.expr)
        print(treestr.ljust(8))


def build_tree(isr_expression, matrix=Operator("M")):
    """Build a tree structure to define response vectors for evaluating the ADC/ISR formulation of a molecular property.
    
    Parameters
    ----------
    isr_expression: <class 'sympy.core.add.Add'> or <class 'sympy.core.mul.Mul'>
        SymPy expression of the ADC/ISR formulation.

    matrix: <class 'sympy.physics.quantum.operator.Operator'>, optional
        The matrix contained in the SymPy expression.

    Returns
    ----------
    tuple
        The first entry is the root expression, i.e., a SymPy expression that contains instances
        of <class 'responsetree.response_operators.ResponseVector'>;
        the second entry is a dictionary with tuples as keys specifying the response vectors.
    """
    root = IsrTreeNode(isr_expression)
    print(type(root))
    build_branches(root, matrix)
    show_tree(root)
    rvecs = {}
    no = 1
    for leaf in root.leaves:
        if isinstance(leaf, ResponseNode):
            old_expr = leaf.expr
            oper_rhs = leaf.rhs
            if isinstance(leaf.rhs, adjoint):
                oper_rhs = leaf.rhs.args[0]
            
            if isinstance(oper_rhs, Mul):
                if isinstance(oper_rhs.args[0], S2S_MTM):
                    key = ((type(oper_rhs.args[0]), oper_rhs.args[1]), leaf.w, leaf.gamma)
                    comp = oper_rhs.args[0].comp
                elif isinstance(oper_rhs.args[1], S2S_MTM):
                    key = ((oper_rhs.args[0], type(oper_rhs.args[1])), leaf.w, leaf.gamma)
                    comp = oper_rhs.args[1].comp
                else:
                    raise ValueError()
            else:
                key = (type(oper_rhs), leaf.w, leaf.gamma)
                comp = oper_rhs.comp

            if key not in rvecs:
                rvecs[key] = {comp: ResponseVector(comp, no)}
                no += 1
            else:
                if comp not in rvecs[key].keys():
                    rv_no = list(rvecs[key].values())[0].no
                    rvecs[key][comp] = ResponseVector(comp, rv_no)
            
            if oper_rhs == leaf.rhs:
                leaf.expr = rvecs[key][comp]
            else:
                leaf.expr = adjoint(rvecs[key][comp])
            traverse_branches(leaf.parent, old_expr, leaf.expr)
    show_tree(root)
    print(rvecs)
    return root.expr, rvecs

if __name__ == "__main__":
    alpha_like = adjoint(F_A) * (M - w - 1j*gamma)**-1 * F_B + adjoint(F_B) * (M + w +  1j*gamma)**-1 * F_A
    beta_like = adjoint(F_A) * (M - w)**-1 * B_B * (M + w)**-1 * F_C
    beta_real = (
        adjoint(F_A) * (M - w_o)**-1 * B_B * (M - w_2)**-1 * F_C
        + adjoint(F_A) * (M - w_o)**-1 * B_C * (M - w_1)**-1 * F_B
        + adjoint(F_C) * (M + w_2)**-1 * B_B * (M + w_o)**-1 * F_A
        + adjoint(F_B) * (M + w_1)**-1 * B_C * (M + w_o)**-1 * F_A
        + adjoint(F_B) * (M + w_1)**-1 * B_A * (M - w_2)**-1 * F_C
        + adjoint(F_C) * (M + w_2)**-1 * B_A * (M - w_1)**-1 * F_B
    )
    gamma_like = adjoint(F_A) * (M - w)**-1 * B_B * (M + w)**-1 * B_D * (M + 2*w)**-1 * F_C
    #build_tree(alpha_like)
    #build_tree(beta_like)
    #build_tree(beta_real)
    #build_tree(gamma_like)
