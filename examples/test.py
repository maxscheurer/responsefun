import numpy as np
from sympy.physics.quantum.state import Bra, Ket, StateBase
from sympy.physics.quantum.operator import HermitianOperator
import sympy.physics.quantum.operator as qmoperator
from sympy import Symbol, Mul, Add, Pow, symbols, adjoint, latex

from anytree import NodeMixin, RenderTree


print("Hallo Max!")


class MTM(qmoperator.Operator):
    pass

class Matrix(qmoperator.Operator):
    pass


class IsrTreeNode(NodeMixin):  # Add Node feature
    def __init__(self, expr, parent=None, children=None):
        super().__init__()
        self.expr = expr
        self.parent = parent
        if children:
            self.children = children


class LRTreeNode(NodeMixin):  # Add Node feature
    def __init__(self, expr, parent=None, children=None):
        super().__init__()
        self.expr = expr
        self.parent = parent
        if children:
            self.children = children


def preorder_traversal(expr):
    # print(expr.func)
    types = [type(e) for e in expr.args]
    # print(types)
    if isinstance(expr, Mul):
        print("multiplication found", expr.args)
    for e in expr.args:
        preorder_traversal(e)


F_alpha = MTM(r"F(\mu_{\alpha})")  # \alpha = {x, y, z}
F_beta = MTM(r"F(\mu_{\beta})")  # \beta = {x, y, z}

X = qmoperator.Operator(r"X_\alpha")
M = Matrix("M")
w = Symbol("w", real=True)

test = F_alpha * (M - w)**(-1) * F_beta + F_beta * (M + w)**(-1) * F_alpha

test2 = test.subs({F_alpha * (M - w)**(-1): X})
print(test2)

# expr_unmodified = F (M-w)^-1 B (M+w)^-1 F
# X B Y
# child_mapping = [X, Y]

# test_AB = child1 + child2 --> addition AB + BA

# child1 = child11_A * F_B --> dot product
# child11 = F_alpha * (M - w)**(-1) --> solve

# child2 = child22 * F_alpha --> dot product
# child22 = F_beta * (M + w)**(-1) --> solve


def build_tree(isr_expression):
    # check what type of operation is needed in isr_expression
    # build top-level node
    pass


root = IsrTreeNode(test)
child1 = IsrTreeNode(F_alpha * (M - w)**(-1) * F_beta, parent=root)
child2 = IsrTreeNode(F_beta * (M + w)**(-1) * F_alpha, parent=root)

child11 = LRTreeNode(F_alpha * (M - w)**(-1), parent=child1)
child22 = LRTreeNode(F_beta * (M + w)**(-1), parent=child2)

for pre, _, node in RenderTree(root):
    treestr = u"%s%s" % (pre, node.expr)
    print(treestr.ljust(8))

print(root.children)

# preorder_traversal(test)