import ast
import textwrap
import inspect


class FunctionSummary:
    def __init__(self, name, params):
        self.name = name
        self.params = set(params)

        self.read_vars = set()
        self.written_vars = set()

        self.read_attrs = dict()
        self.written_attrs = dict()

        self.calls = set()

        # Which parameters are mutated via attribute/subscript write
        # self.mutated_params = set()

    # def merge_from(self, other):
    #    changed = False
    #
    #    for attr in (
    #        "read_vars",
    #        "written_vars",
    #        "read_attrs",
    #        "written_attrs",
    #        "mutated_params",
    #    ):
    #        before = getattr(self, attr).copy()
    #        getattr(self, attr).update(getattr(other, attr))
    #        if before != getattr(self, attr):
    #            changed = True
    #
    #    return changed

    def __str__(self) -> str:
        return f"""
Summary:
  fun: {self.name}
  params: {self.params}
  ----
  read vars: {self.read_vars}
  write vars: {self.written_vars}
  read attrs: {self.read_attrs}
  write attrs: {self.written_attrs}
  ----
  calls: {self.calls}
"""


#
# def set_attribute(attrs: dict, resolved: list[str]) -> None:
#     assert len(resolved) >= 2
#     key = resolved[0]
#     if len(resolved) > 2:
#         lst = attrs.setdefault(key, [])
#         S = {}
#         lst.append(S)
#         set_attribute(S, resolved[1:])
#     elif len(resolved) == 2:
#         lst = attrs.setdefault(key, [])
#         if resolved[1] not in lst:
#             lst.append(resolved[1])
#     else:
#         raise RuntimeError("Invalid resolved attribute or a bug!")
#


def set_attribute(attrs: dict, resolved: list[str]) -> None:
    assert len(resolved) >= 1
    key = resolved[0]
    nxt_attrs = attrs.setdefault(key, {})
    if len(resolved) > 1:
        set_attribute(nxt_attrs, resolved[1:])


class AccessAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.functions = {}
        self.current = None

    def analyze(self, obj):
        source = textwrap.dedent(inspect.getsource(obj))
        tree = ast.parse(source)

        print("TREE", ast.dump(tree, indent=1))

        self.visit(tree)

        # propagate_effects(self.functions)

        return self.functions

    # ----------------------------
    # Function definitions
    # ----------------------------

    def visit_FunctionDef(self, node):
        params = [arg.arg for arg in node.args.args]
        summary = FunctionSummary(node.name, params)
        self.functions[node.name] = summary

        prev = self.current
        self.current = summary

        for stmt in node.body:
            self.visit(stmt)

        self.current = prev

    # ----------------------------
    # Variable access
    # ----------------------------

    def visit_Name(self, node):
        if not self.current:
            return

        print("Name", ast.dump(node, indent=1))
        if isinstance(node.ctx, ast.Load):
            self.current.read_vars.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.current.written_vars.add(node.id)

    # ----------------------------
    # Attribute access
    # ----------------------------

    def visit_Attribute(self, node):
        if not self.current:
            return

        full: list[str] = self._resolve_attribute(node)

        if isinstance(node.ctx, ast.Load):
            set_attribute(self.current.read_attrs, full)
        elif isinstance(node.ctx, ast.Store):
            set_attribute(self.current.written_attrs, full)

            # detect parameter mutation
        # base = self._root_name(node)
        # if base in self.current.params:
        #    self.current.mutated_params.add(base)

        self.generic_visit(node)

    # ----------------------------
    # Subscript writes (x[0] = ...)
    # ----------------------------

    def visit_Subscript(self, node):
        if not self.current:
            return

        # if isinstance(node.ctx, ast.Store):
        #    base = self._root_name(node)
        #    if base in self.current.params:
        #        self.current.mutated_params.add(base)

        self.generic_visit(node)

    # ----------------------------
    # Calls
    # ----------------------------

    def visit_Call(self, node):
        if not self.current:
            return

        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                # self.method() — record as a call, not an attribute read
                self.current.calls.add(node.func.attr)
                for arg in node.args:
                    self.visit(arg)
                return
        elif isinstance(node.func, ast.Name):
            self.current.calls.add(node.func.id)

        self.generic_visit(node)

    # ----------------------------
    # Helpers
    # ----------------------------

    def _root_name(self, node):
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _resolve_attribute(self, node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return list(reversed(parts))


def propagate_effects(functions):
    """
    Fixpoint propagation across call graph.
    """
    changed = True
    while changed:
        changed = False
        for fn in functions.values():
            for callee_name in fn.calls:
                if callee_name in functions:
                    callee = functions[callee_name]
                    if fn.merge_from(callee):
                        changed = True
