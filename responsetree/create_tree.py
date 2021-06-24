from sympy.physics.quantum.state import Bra, Ket, StateBase
from sympy.physics.quantum.operator import HermitianOperator
import sympy.physics.quantum.operator as qmoperator
from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex

from anytree import NodeMixin, RenderTree

from itertools import permutations

from responsetree.response_operators import MTM, S2S_MTM, ResponseVector, Matrix


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
                    rvecs[key] = ResponseVector(comp, no)
                    no += 1
                leaf.expr = adjoint(rvecs[key])
            else:
                comp = leaf.rhs.comp
                key = (leaf.rhs, leaf.w, leaf.gamma)
                if key not in rvecs:
                    rvecs[key] = ResponseVector(comp, no)
                    no += 1
                leaf.expr = rvecs[key]
            traverse_branches(leaf.parent, old_expr, leaf.expr)
    show_tree(root)
    print(rvecs)


w, gamma, w_o, w_1, w_2 = symbols(r"w, \gamma, w_{\sigma}, w_{1}, w_{2}", real=True)
F_A = MTM("A") # A = {x, y, z}
F_B = MTM("B") # B = {x, y, z}
F_C = MTM("C") # C = {x, y, z}
F_D = MTM("D") # D = {x, y, z}
B_A = S2S_MTM("A") # A = {x, y, z}
B_B = S2S_MTM("B") # B = {x, y, z}
B_C = S2S_MTM("C") # C = {x, y, z}
B_D = S2S_MTM("D") # D = {x, y, z}
M = Matrix("M")

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
#build_tree(alpha_like, M)
#build_tree(beta_like, M)
build_tree(beta_real, M)
#build_tree(gamma_like, M)

# generate equation for beta via permutation
perms = list(permutations([("A", -w_o), ("B", w_1), ("C", w_2)]))

F1 = qmoperator.Operator("F1")
B2 = qmoperator.Operator("B2")
F3 = qmoperator.Operator("F3")
wF1, wF3 = symbols("wF1, wF3")

beta_term = adjoint(F1) * (M + wF1)**-1 * B2 * (M - wF3)**-1 * F3
beta_real2 = 0 

for p in perms:
    subs_list = [
        (F1, MTM(p[0][0])),
        (wF1, p[0][1]),
        (B2, S2S_MTM(p[1][0])),
        (wF3, p[2][1]),
        (F3, MTM(p[2][0]))
    ]
    beta_real2 += beta_term.subs(subs_list)
    
#print(beta_real2)
#print(beta_real==beta_real2)
