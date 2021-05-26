from sympy.physics.quantum.state import Bra, Ket, StateBase
from sympy.physics.quantum.operator import HermitianOperator
import sympy.physics.quantum.operator as qmoperator
from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex

from anytree import NodeMixin, RenderTree

class MTM(qmoperator.Operator):
    pass

class Matrix(qmoperator.Operator):
    pass

class IsrTreeNode(NodeMixin):
    def __init__(self, expr, parent=None, children=None):
        super().__init__()
        self.expr = expr
        self.parent = parent
        if children:
            self.children = children
 
class ResponseNode(NodeMixin):
    def __init__(self, expr, tinv, rhs, parent=None, children=None):
        super().__init__()
        self.expr = expr
        self.tinv = tinv
        self.rhs = rhs #rhs of response equation
        self.w = tinv.subs([(M, 0), (gamma, 0)])
        self.gamma = tinv.subs([(M, 0), (w, 0)])
        self.parent = parent
        if children:
            self.children = children

class ResponseVector(qmoperator.Operator):
    pass

def acceptable_rhs_lhs(term):
    if isinstance(term, adjoint):
        op_expr = term.args[0]
    else:
        op_expr = term
    return isinstance(op_expr, MTM)

def build_branches(node):
    if isinstance(node.expr, Add):
        node.children = [IsrTreeNode(term) for term in node.expr.args]
        for child in node.children:
            build_branches(child)
    elif isinstance(node.expr, Mul):
        children = []
        for i, term in enumerate(node.expr.args):
            if isinstance(term, Pow) and term.args[1] == -1 and M in term.args[0].args:
                tinv = term.args[0]
                lhs = node.expr.args[i-1]
                rhs = node.expr.args[i+1]
                if acceptable_rhs_lhs(rhs):
                    children.append(ResponseNode(tinv**-1 * rhs, tinv, rhs))
                elif acceptable_rhs_lhs(lhs):
                    children.append(ResponseNode(lhs * tinv**-1, tinv, lhs))
                else:
                    print("No invertable term found")
        node.children = children
                    
def traverse_branches(node, old_expr, new_expr):
    oe = node.expr
    ne = node.expr.subs(old_expr, new_expr)
    node.expr = ne
    if not node.is_root:
        traverse_branches(node.parent, oe, ne)

def show_tree(root):
    for pre, _, node in RenderTree(root):
        treestr = u"%s%s" % (pre, node.expr)
        print(treestr.ljust(8))

def build_tree(isr_expression):
    root = IsrTreeNode(isr_expression)
    build_branches(root)
    show_tree(root)
    for leaf in root.leaves:
        if isinstance(leaf, ResponseNode):
            old_expr = leaf.expr
            if isinstance(leaf.rhs, adjoint):
                comp = str(leaf.rhs.args[0])[3]
                leaf.expr = adjoint(ResponseVector(r"X_{"+comp+"}"))
            else:
                comp = str(leaf.rhs)[3] 
                leaf.expr = ResponseVector(r"X_{"+comp+"}")
            traverse_branches(leaf.parent, old_expr, leaf.expr)
    show_tree(root)

w, gamma = symbols("w, \gamma", real=True)
F_A = MTM(r"F_{A}") # A = {x, y, z}
F_B = MTM(r"F_{B}") # B = {x, y, z}
F_C = MTM(r"F_{C}") # C = {x, y, z}
B_B = qmoperator.Operator(r"B_{B}") # B = {x, y, z}
B_D = qmoperator.Operator(r"B_{D}") # D = {x, y, z}
M = Matrix("M")

alpha = adjoint(F_A) * (M - w - 1j*gamma)**-1 * F_B + adjoint(F_B) * (M + w +  1j*gamma)**-1 * F_A
beta = adjoint(F_A) * (M - w)**-1 * B_B * (M + w)**-1 * F_C
gamma = adjoint(F_A) * (M - w)**-1 * B_B * (M + w)**-1 * B_D * (M + 2*w)**-1 * F_C
build_tree(alpha)
#build_tree(beta)
#build_tree(gamma)