"""
Session-scoped fixtures that prepare the tests/lean/ lake project before any
test in this directory runs:

1. ``sync_core_templates`` — copies template files into Core/ (same as before).
2. ``generate_lean_files`` — generates:
     • ZerothHammer.lean  (the zeroth_hammer tactic, standalone)
     • Certs/Countdown.lean  (self-contained certificate, inline module)
     • Certs/TwoVars.lean
     • Certs/Collatz.lean
     • Certs/ScalarEnc*.lean  (functional + scalar encodings)
     • Certs/RelEnc*/     (those two + ScalarRel, one module each)
     • Certs/ArgmaxScalar.lean  (the scalar Argmax variants)

The generated files import ZerothHammer for the tactic and define their own
module-specific ``simp_mat`` / ``simp_defs`` / ``mat_collapse`` macros.
"""
import shutil
from pathlib import Path

import pytest
import torch

from zrth import Module, Wire, Term, LIA, Int, Bool, Var, X
from zrth.analyzer import convert_method
from zrth.lean.cert import CertificateData, generate_zeroth_hammer_lean, smt_predicates_to_lean
from zrth.lean.project import CORE_FILES, TEMPLATE_DIR, generate_standalone_cert_lean
from zrth.lean.translate.scalar import _argmax_scalar_def_lines

_LEAN_DIR = Path(__file__).parent
_CORE_DIR = _LEAN_DIR / "Core"
_CERTS_DIR = _LEAN_DIR / "Certs"


# ──────────────────────────────────────────────────────────────
# Module factories (same modules as test_lean_svcomp.py)
# ──────────────────────────────────────────────────────────────


def _make_countdown() -> Module:
    def init():
        return 100

    def update(old_x):
        if old_x == 0:
            return 100
        return old_x - 1

    s = Var(Int([1, 1]))
    return Module.sequential(
        [s],
        convert_method(init, {}, [X(s)], theory=LIA),
        convert_method(update, {"old_x": s}, [X(s)], theory=LIA),
    )


def _make_twovars() -> Module:
    def init():
        return 0, 10

    def update(old_x, old_y):
        if old_x < old_y:
            return old_x + 1, old_y
        return 0, 10

    x = Var(Int([1, 1]))
    y = Var(Int([1, 1]))
    return Module.sequential(
        [x, y],
        convert_method(init, {}, [X(x), X(y)], theory=LIA),
        convert_method(update, {"old_x": x, "old_y": y}, [X(x), X(y)], theory=LIA),
    )


def _make_collatz() -> Module:
    def init():
        return 7

    def update(old_x):
        if old_x == 1:
            return 7
        if old_x > 4:
            return old_x - 3
        if old_x > 1:
            return old_x - 1
        return old_x

    s = Var(Int([1, 1]))
    return Module.sequential(
        [s],
        convert_method(init, {}, [X(s)], theory=LIA),
        convert_method(update, {"old_x": s}, [X(s)], theory=LIA),
    )


def _make_counter() -> Module:
    """3×1 vector-state counter with LIA.Linear transitions (matrix cert).

    state = (x, y, z); init = (0, y0, z0); update increments x while x < y or
    x < z, else resets x to 0. Exercises the Linear affine map plus tuple-select
    (`s[i][j]`) predicates end-to-end through zeroth_hammer.
    """
    state = Var(Int([3, 1]))
    extl = Var(Int([2, 1]))
    zero31 = torch.zeros((3, 1), dtype=torch.int64)
    zero11 = torch.zeros((1, 1), dtype=torch.int64)

    A = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)
    init = [Term(LIA.Linear(A, zero31), [X(state)], [X(extl)])]

    x, y, z = Wire(Int([1, 1])), Wire(Int([1, 1])), Wire(Int([1, 1]))
    x_lt_y, x_lt_z, cond = Wire(Bool([1, 1])), Wire(Bool([1, 1])), Wire(Bool([1, 1]))
    result_true, result_false = Wire(Int([3, 1])), Wire(Int([3, 1]))
    row_x = torch.tensor([[1, 0, 0]], dtype=torch.int64)
    row_y = torch.tensor([[0, 1, 0]], dtype=torch.int64)
    row_z = torch.tensor([[0, 0, 1]], dtype=torch.int64)
    e1 = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    diag_yz = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.int64)
    update = [
        Term(LIA.Linear(row_x, zero11), [x], [state]),
        Term(LIA.Linear(row_y, zero11), [y], [state]),
        Term(LIA.Linear(row_z, zero11), [z], [state]),
        Term(LIA.Lt(), [x_lt_y], [x, y]),
        Term(LIA.Lt(), [x_lt_z], [x, z]),
        Term(LIA.Or(), [cond], [x_lt_y, x_lt_z]),
        Term(LIA.Linear(torch.eye(3, dtype=torch.int64), e1), [result_true], [state]),
        Term(LIA.Linear(diag_yz, zero31), [result_false], [state]),
        Term(LIA.Ite(), [X(state)], [cond, result_true, result_false]),
    ]
    return Module.sequential([state, extl], init, update)


def _make_countdown_vec(n: int) -> Module:
    """Countdown embedded in an n-vector state (only s[0] is live; s[1..n-1] carried).

    Every transition is an n-wide MatMul contraction (row·s, Iₙ·s, diag·s). Because
    the codegen pre-contracts constant Linear ops (each output element becomes an
    explicit O(nnz) linear combination), the generated defs and proofs stay small
    and `omega`-friendly regardless of n — this is the scaling regression test.
    """
    s = Var(Int([n, 1]))
    z11 = torch.zeros((1, 1), dtype=torch.int64)

    init_vec = torch.zeros((n, 1), dtype=torch.int64)
    init_vec[0][0] = 100
    init = [Term(LIA.Int(init_vec), [X(s)])]

    x = Wire(Int([1, 1]))
    zc = Wire(Int([1, 1]))
    cond = Wire(Bool([1, 1]))
    reset = Wire(Int([n, 1]))
    dec = Wire(Int([n, 1]))

    row0 = torch.zeros((1, n), dtype=torch.int64)
    row0[0][0] = 1  # 1×n: extract s[0]
    diag_keep = torch.eye(n, dtype=torch.int64)
    diag_keep[0][0] = 0  # zero s[0], keep the rest
    b100 = torch.zeros((n, 1), dtype=torch.int64)
    b100[0][0] = 100
    In = torch.eye(n, dtype=torch.int64)
    bneg = torch.zeros((n, 1), dtype=torch.int64)
    bneg[0][0] = -1

    update = [
        Term(LIA.Linear(row0, z11), [x], [s]),            # x = s[0]
        Term(LIA.Int(torch.tensor([[0]])), [zc]),
        Term(LIA.Eq(), [cond], [x, zc]),                  # cond = (x == 0)
        Term(LIA.Linear(diag_keep, b100), [reset], [s]),  # reset: s[0] := 100
        Term(LIA.Linear(In, bneg), [dec], [s]),           # dec:   s[0] := s[0] - 1
        Term(LIA.Ite(), [X(s)], [cond, reset, dec]),
    ]
    return Module.sequential([s], init, update)


def _make_mixed() -> Module:
    """ctrl = [x : 1x1, v : 3x1] — the state whose wires and elements disagree.

    Four state elements against two ctrl wires, so wire 1's data starts at
    flat slot 1. Everything else here has either all-1x1 wires (slot i *is*
    wire i) or a single wire (the slice is the whole tuple), and neither
    catches a per-wire projection of a per-element tuple. Both transitions
    read both wires so the slices cannot be independently wrong.
    """
    x = Var(Int([1, 1]))
    v = Var(Int([3, 1]))
    z11 = torch.zeros((1, 1), dtype=torch.int64)
    v0 = torch.tensor([[1], [2], [3]], dtype=torch.int64)

    init = [
        Term(LIA.Int(torch.zeros((1, 1), dtype=torch.int64)), [X(x)]),
        Term(LIA.Int(v0), [X(v)]),
    ]

    head, zc = Wire(Int([1, 1])), Wire(Int([1, 1]))
    cond = Wire(Bool([1, 1]))
    reset, dec = Wire(Int([3, 1])), Wire(Int([3, 1]))
    row0 = torch.tensor([[1, 0, 0]], dtype=torch.int64)
    zero33 = torch.zeros((3, 3), dtype=torch.int64)
    bneg = torch.tensor([[-1], [0], [0]], dtype=torch.int64)

    update = [
        Term(LIA.Linear(row0, z11), [head], [v]),          # head = v[0]
        Term(LIA.Add(), [X(x)], [x, head]),                # x' = x + v[0]
        Term(LIA.Int(torch.zeros((1, 1), dtype=torch.int64)), [zc]),
        Term(LIA.Eq(), [cond], [head, zc]),                # cond = (v[0] == 0)
        Term(LIA.Linear(zero33, v0), [reset], [v]),        # reset: v := (1,2,3)
        Term(LIA.Linear(torch.eye(3, dtype=torch.int64), bneg), [dec], [v]),
        Term(LIA.Ite(), [X(v)], [cond, reset, dec]),
    ]
    return Module.sequential([x, v], init, update)


_COUNTDOWN_CERT = CertificateData(
    prp="s[0][0] == 0",
    inv="And(s[0][0] >= 0, s[0][0] <= 100)",
    ranking="Ite(s[0][0] == 0, 0, s[0][0])",
)


# (elem_ty, n) pairs for the generated scalar Argmax variants. n = 1 is the
# degenerate single-element case; the rest exercise the fold.
# Int and Real at the same width sit together on purpose: the variant
# names used to be built from `n` alone, so these two collided.
# Widths that exercise the scalar encoding's flattening: all-1x1, then
# multi-element state, up to the 32 that stresses the index enumeration.
_SCALAR_ENC_SPECS = [
    ("Scalar", _make_countdown),
    ("Vec6", lambda: _make_countdown_vec(6)),
    ("Counter", _make_counter),
    ("Vec32", lambda: _make_countdown_vec(32)),
]


# The relational encodings sit on top of the scalar one, so they need every
# width the scalar specs cover, plus the two multi-wire states: `TwoVars`,
# where each wire is one element so wire index and slot index coincide (what
# the per-wire projection assumed), and `Mixed`, where they do not.
_REL_ENC_SPECS = _SCALAR_ENC_SPECS + [
    ("TwoVars", _make_twovars),
    ("Mixed", _make_mixed),
]


_ARGMAX_SCALAR_SPECS = [
    ("Int", 1),
    ("Int", 2),
    ("Int", 4),
    ("Real", 3),
    ("Real", 4),
]


# (name, module, safety property) for the `--fbk-proveit` equivalence proof.
# One per shape class the route accepts: a scalar Int state, a mixed Bool/Int
# state, and a module whose transition is a net (`Linear` -> `ReLU` ->
# `Linear`), which is the case that needs the matrix simp set.
_BRIDGE_SPECS = [
    ("Countdown", _make_countdown, "(and (>= s0 0) (<= s0 100))"),
    ("TwoVars", _make_twovars, "(and (>= s0 0) (<= s0 s1) (= s1 10))"),
]


_CERT_SPECS = [
    (
        "BigCounter",
        lambda: _make_countdown_vec(6),
        _COUNTDOWN_CERT,
    ),
    (
        "HugeCounter",
        lambda: _make_countdown_vec(32),
        _COUNTDOWN_CERT,
    ),
    (
        "Counter",
        _make_counter,
        CertificateData(
            prp="s[0][0] == 0",
            inv="And(s[0][0] >= 0, Or(s[0][0] <= s[1][0], s[0][0] <= s[2][0]))",
            init_pre="And(e[0][0] >= 0, e[1][0] >= 0)",
            ranking="Ite(s[0][0] == 0, 0, (Ite(s[1][0] >= s[2][0], s[1][0], s[2][0]) - s[0][0]) + 1)",
        ),
    ),
    (
        "Countdown",
        _make_countdown,
        CertificateData(
            prp="(= s0 0)",
            inv="(and (>= s0 0) (<= s0 100))",
            ranking="(ite (= s0 0) 0 s0)",
        ),
    ),
    (
        # The safety shape: `rule_globally`, an invariant that implies the
        # property, and no ranking function anywhere in the file. Nothing
        # else in this suite compiles a `--safety` certificate.
        "CountdownSafe",
        _make_countdown,
        CertificateData(
            kind="safety",
            prp="(<= s0 100)",
            inv="(and (>= s0 0) (<= s0 100))",
        ),
    ),
    (
        "TwoVars",
        _make_twovars,
        CertificateData(
            prp="(= s0 s1)",
            inv="(and (>= s0 0) (<= s0 s1) (= s1 10))",
            ranking="(ite (= s0 s1) 0 (- s1 s0))",
        ),
    ),
    (
        "Collatz",
        _make_collatz,
        CertificateData(
            prp="(= s0 1)",
            inv="(and (>= s0 1) (<= s0 8))",
            ranking="(ite (= s0 1) 0 (- s0 1))",
        ),
    ),
]


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def sync_core_templates() -> None:
    """Copy the current template files into Core/ before the test session."""
    _CORE_DIR.mkdir(exist_ok=True)
    for name in CORE_FILES:
        src = TEMPLATE_DIR / "Core" / name
        dst = _CORE_DIR / name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


@pytest.fixture(scope="session", autouse=True)
def generate_lean_files(sync_core_templates) -> None:
    """Generate ZerothHammer.lean and Certs/*.lean for the lake project."""
    # ZerothHammer.lean — standalone tactic
    hammer_file = _LEAN_DIR / "ZerothHammer.lean"
    hammer_file.write_text(generate_zeroth_hammer_lean())

    # Certs/ — one self-contained certificate per module + scalar encoding file
    _CERTS_DIR.mkdir(exist_ok=True)
    for name, make_module, cert_data in _CERT_SPECS:
        module = make_module()
        lean_cert = smt_predicates_to_lean(cert_data, module)
        content = generate_standalone_cert_lean(module, lean_cert)
        (_CERTS_DIR / f"{name}.lean").write_text(content)

    # Certs/ArgmaxScalar.lean — the scalar Argmax variants exactly as the
    # generator emits them. Each carries an `argmax1d_scalar_n_eq` theorem
    # proving it equal to Core.Mat.argmax_1d, so building this file is what
    # keeps the unrolled scalar form and the matrix definition in step. No
    # certificate module currently uses Argmax, so nothing else compiles it.
    # Certs/ScalarEnc.lean — the functional and scalar encodings together, the
    # way a generated project arranges them (System/System.lean +
    # System/Scalar.lean). The `_scalar_eq` theorems only elaborate if the
    # flattening lines up, and no certificate carries a Scalar section, so
    # this is the only thing that compiles them.
    from zrth.lean.translate import ModuleToLean4

    for name, make_module in _SCALAR_ENC_SPECS:
        t = ModuleToLean4(make_module())
        (_CERTS_DIR / f"ScalarEnc{name}.lean").write_text(
            "import Core.Mat\nimport Core.Box\n\n"
            + t.to_lean_functional()
            + "\n\n"
            + t.to_lean_scalar()
            + "\n"
        )

    # Certs/RelEnc*/ — functional, scalar and ScalarRel, one module each,
    # exactly the way `create_project` splits them: System/Scalar.lean imports
    # System/System.lean and System/ScalarRel.lean imports Scalar.
    # `ScalarRel.effect_i` is per wire while the state tuple it is compared to
    # is per element, so these are what keep the slice and the codomain in
    # agreement; no certificate carries a ScalarRel section.
    #
    # The split is load-bearing, not cosmetic. These used to be concatenated
    # into one file, which is a weaker test than separate modules: a `match`
    # elaborates to an auxiliary matcher that is reused within a module but
    # regenerated across module boundaries, so an `effect_i_eq` saw one
    # matcher constant on both sides here and two different ones in a real
    # project, and every generated project with a multi-element ctrl wire
    # failed to compile while this test passed.
    for name, make_module in _REL_ENC_SPECS:
        t = ModuleToLean4(make_module())
        base = f"Certs.RelEnc{name}"
        d = _CERTS_DIR / f"RelEnc{name}"
        d.mkdir(exist_ok=True)
        (d / "Base.lean").write_text(
            "import Core.Mat\nimport Core.Box\n\n" + t.to_lean_functional() + "\n"
        )
        (d / "Scalar.lean").write_text(
            f"import Core.Basic\nimport {base}.Base\n\n" + t.to_lean_scalar() + "\n"
        )
        (d / "ScalarRel.lean").write_text(
            f"import Core.Basic\nimport {base}.Scalar\n\n" + t.to_lean_rel() + "\n"
        )
        (_CERTS_DIR / f"RelEnc{name}.lean").write_text(f"import {base}.ScalarRel\n")

    # Certs/Bridge*.lean — the `--fbk-proveit` route's model, the module's
    # own encodings, and the proof that the first is the second. The bridge
    # normally imports `System.*` and the NA model from a generated project;
    # inlining all of them in the order the project has them is the same
    # elaboration problem, and it runs in this warm build instead of a cold
    # `lake update` per case.
    from zrth.lean.cert import generate_data_lean
    from zrth.lean.common import LeanContext
    from zrth.lean.fbk_proveit import property_to_bool_lean
    from zrth.lean.translate.fbk import atom_to_lean_na
    from zrth.lean.translate.fbk_bridge import atom_to_lean_fbk_bridge

    def _strip(src: str) -> str:
        out, in_doc = [], False
        for line in src.splitlines():
            if line.startswith("/-"):
                in_doc = True
            if in_doc:
                if "-/" in line:
                    in_doc = False
                continue
            if line.startswith("import "):
                continue
            out.append(line)
        return "\n".join(out)

    for name, make_module, prop in _BRIDGE_SPECS:
        module = make_module()
        ctx = LeanContext(module)
        t = ModuleToLean4(module)
        cert = smt_predicates_to_lean(CertificateData(prp=prop, kind="safety"), module)
        (_CERTS_DIR / f"Bridge{name}.lean").write_text(
            "\n".join([
                "import Core.Basic", "import Core.Box", "import Core.Mat",
                "import Smt", "",
                t.to_lean_functional(), "",
                t.to_lean_scalar(), "",
                _strip(generate_data_lean(ctx, cert)), "",
                _strip(atom_to_lean_na(ctx, property_to_bool_lean(module, prop),
                                       module_name=name)), "",
                _strip(atom_to_lean_fbk_bridge(ctx, na_module=name)),
            ]) + "\n"
        )

    argmax_lines = ["import Core.Mat", ""]
    for elem_ty, n in _ARGMAX_SCALAR_SPECS:
        argmax_lines.extend(_argmax_scalar_def_lines(elem_ty, n))
    (_CERTS_DIR / "ArgmaxScalar.lean").write_text("\n".join(argmax_lines))
