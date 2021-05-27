from sympy.physics.quantum.state import Bra, Ket, StateBase
from sympy.physics.quantum.operator import HermitianOperator
import sympy.physics.quantum.operator as qmoperator
from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex

from anytree import NodeMixin, RenderTree

from itertools import permutations

class MTM(qmoperator.Operator):
    def __init__(self, expr):
        super().__init__()
        self.expr = expr

    @property
    def comp(self):
        expr_list = self.expr.strip()
        if "{" in expr_list:
            i_comp = expr_list.index("{") + 1
        else:
            raise ValueError("Expression does not contain { to specify the component.")
        return expr_list[i_comp]

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
        self.gamma = tinv.subs([(M, 0), (self.w, 0)])
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

def build_branches(node, matrix):
    if isinstance(node.expr, Add):
        node.children = [IsrTreeNode(term) for term in node.expr.args]
        for child in node.children:
            build_branches(child, matrix)
    elif isinstance(node.expr, Mul):
        children = []
        for i, term in enumerate(node.expr.args):
            if isinstance(term, Pow) and term.args[1] == -1 and matrix in term.args[0].args:
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

def build_tree(isr_expression, matrix):
    root = IsrTreeNode(isr_expression)
    build_branches(root, matrix)
    show_tree(root)
    rvecs = {}
    no = 1
    for leaf in root.leaves:
        if isinstance(leaf, ResponseNode):
            old_expr = leaf.expr
            if isinstance(leaf.rhs, adjoint):
                comp = leaf.rhs.args[0].comp
                key = (leaf.rhs.args[0], leaf.w, leaf.gamma)
                if key not in rvecs:
                    rvecs[key] = ResponseVector(r"X_{"+comp+"}", no)
                    no += 1
                leaf.expr = adjoint(rvecs[key])
            else:
                comp = leaf.rhs.comp
                key = (leaf.rhs, leaf.w, leaf.gamma)
                if key not in rvecs:
                    rvecs[key] = ResponseVector(r"X_{"+comp+"}", no)
                    no += 1
                leaf.expr = rvecs[key]
            traverse_branches(leaf.parent, old_expr, leaf.expr)
    show_tree(root)
    print(rvecs)

w, gamma, wo, w1, w2 = symbols(r"w, \gamma, w_{\sigma}, w_{1}, w_{2}", real=True)
F_A = MTM(r"F_{A}") # A = {x, y, z}
F_B = MTM(r"F_{B}") # B = {x, y, z}
F_C = MTM(r"F_{C}") # C = {x, y, z}
B_A = qmoperator.Operator(r"B_{A}") # A = {x, y, z}
B_B = qmoperator.Operator(r"B_{B}") # B = {x, y, z}
B_C = qmoperator.Operator(r"B_{C}") # C = {x, y, z}
B_D = qmoperator.Operator(r"B_{D}") # D = {x, y, z}
M = Matrix("M")

alpha_like = adjoint(F_A) * (M - w - 1j*gamma)**-1 * F_B + adjoint(F_B) * (M + w +  1j*gamma)**-1 * F_A
beta_like = adjoint(F_A) * (M - w)**-1 * B_B * (M + w)**-1 * F_C
beta_real = (
    adjoint(F_A) * (M - wo)**-1 * B_B * (M - w2)**-1 * F_C
    + adjoint(F_A) * (M - wo)**-1 * B_C * (M - w1)**-1 * F_B
    + adjoint(F_C) * (M + w2)**-1 * B_B * (M + wo)**-1 * F_A
    + adjoint(F_B) * (M + w1)**-1 * B_C * (M + wo)**-1 * F_A
    + adjoint(F_B) * (M + w1)**-1 * B_A * (M - w2)**-1 * F_C
    + adjoint(F_C) * (M + w2)**-1 * B_A * (M - w1)**-1 * F_B
)
gamma_like = adjoint(F_A) * (M - w)**-1 * B_B * (M + w)**-1 * B_D * (M + 2*w)**-1 * F_C
#build_tree(alpha_like, M)
#build_tree(beta_like, M)
build_tree(beta_real, M)
#build_tree(gamma_like, M)

#generate equation for beta via permutation
perms = list(permutations([("A", -wo), ("B", w1), ("C", w2)]))

F1 = qmoperator.Operator("F1")
B2 = qmoperator.Operator("B2")
F3 = qmoperator.Operator("F3")
wF1, wF3 = symbols("wF1, wF3")

beta_term = adjoint(F1) * (M + wF1)**-1 * B2 * (M - wF3)**-1 * F3
beta_real2 = 0

for p in perms:
    subs_list = [(F1, MTM(r"F_{"+p[0][0]+"}")), (wF1, p[0][1]), (B2, qmoperator.Operator(r"B_{"+p[1][0]+"}")), (wF3, p[2][1]), (F3, MTM(r"F_{"+p[2][0]+"}"))]
    beta_real2 += beta_term.subs(subs_list)
    
#print(beta_real2)
#print(beta_real==beta_real2)