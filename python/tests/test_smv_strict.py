"""`parse_smv(strict=True)`: the module means what NuSMV means, or the parse
is refused."""

from pathlib import Path

import pytest
import torch

from zrth import X
from zrth.eval import execute_init, execute_update
from zrth.smv import SMVRefused, parse_smv

FIXTURES = Path(__file__).parent / "smv_fixtures"


def _run(text, steps, *, strict=True):
    """Values of every variable after init and after each step in `steps`,
    a list of {input: (value read this round, value latched)}."""
    module, names = parse_smv(text, strict=strict)
    state = {}
    execute_init(state, module.atoms)
    trace = []

    def snapshot():
        trace.append({n: int(state[X(w)].item()) for n, w in names.items()
                      if X(w) in state})
        for v in module.ctrl:
            state[v] = state[X(v)].clone()

    snapshot()
    for inputs in steps:
        for name, (now, latched) in inputs.items():
            state[X(names[name])] = torch.tensor([[now]])
            state[names[name]] = torch.tensor([[latched]])
        execute_update(state, module.atoms)
        snapshot()
    return trace


INVAR_C_D = """MODULE main
IVAR i : boolean;
VAR {first} : unsigned word[4];
    {second} : unsigned word[4];
{define}
ASSIGN
  init(c) := 0ud4_0;
  next(c) := i ? d : c;
"""


@pytest.mark.parametrize("define", ["INVAR d = c + 0ud4_1;",
                                    "ASSIGN d := c + 0ud4_1;"])
@pytest.mark.parametrize("first,second", [("c", "d"), ("d", "c")])
def test_a_value_given_in_every_state_holds_in_every_state(define, first, second):
    text = INVAR_C_D.format(first=first, second=second, define=define)
    trace = _run(text, [{"i": (1, 0)}, {"i": (1, 1)}])
    assert [(s["c"], s["d"]) for s in trace] == [(0, 1), (1, 2), (2, 3)]


def test_an_input_is_read_in_the_round_it_labels():
    text = """MODULE main
IVAR i : boolean;
VAR c : boolean;
ASSIGN init(c) := FALSE; next(c) := i;
"""
    trace = _run(text, [{"i": (1, 0)}, {"i": (0, 1)}])
    assert [s["c"] for s in trace] == [0, 1, 0]


def test_next_of_an_expression_reads_next_values():
    text = """MODULE main
VAR a : unsigned word[4];
    b : unsigned word[4];
ASSIGN
  init(a) := 0ud4_0; next(a) := a + 0ud4_1;
  init(b) := 0ud4_0; next(b) := next(a + 0ud4_1);
"""
    trace = _run(text, [{}])
    assert trace[1] == {"a": 1, "b": 2}


def test_trans_assignments_are_read():
    text = """MODULE main
VAR x : boolean;
INIT x = FALSE
TRANS next(x) = (!x)
"""
    for strict in (True, False):
        assert [s["x"] for s in _run(text, [{}, {}], strict=strict)] == [0, 1, 0]


def test_mod_is_read():
    text = """MODULE main
VAR x : unsigned word[4];
ASSIGN init(x) := 0ud4_7; next(x) := x mod 0ud4_3;
"""
    assert [s["x"] for s in _run(text, [{}])] == [7, 1]


def _model(body, decls="VAR x : boolean;"):
    return f"MODULE main\n{decls}\n{body}\n"


REFUSED = {
    "no next": (_model("ASSIGN init(x) := TRUE;"), "`x` has no next value"),
    "no init": (_model("ASSIGN next(x) := x;"), "`x` has no initial value"),
    "integer with words": (_model("", "VAR x : integer; w : unsigned word[4];"),
                           "`x` has an integer type in a model with words"),
    "range with words": (_model("", "VAR x : 0..9; w : unsigned word[4];"),
                         "`x` has an integer type in a model with words"),
    "numeric enum with words": (_model("", "VAR x : {0, 2}; w : unsigned word[4];"),
                                "`x` has an integer type in a model with words"),
    "mixed enum": (_model("", "VAR x : {idle, 2};"), "mixes names and numbers"),
    "enum clash": (_model("ASSIGN init(x) := TRUE; next(x) := x;",
                          "VAR x : boolean; s : {x, y};"),
                   "`x` is both an enum value"),
    "input in init": (_model("ASSIGN init(x) := i; next(x) := x;",
                             "IVAR i : boolean; VAR x : boolean;"),
                      r"input variable `i` is read .* \(in init\(x\)\)"),
    "input in INVAR": (_model("INVAR x = i",
                              "IVAR i : boolean; VAR x : boolean;"),
                       "input variable `i` is read"),
    "next of input": (_model("ASSIGN init(x) := TRUE; next(x) := next(i);",
                             "IVAR i : boolean; VAR x : boolean;"),
                      "input variable `i` is read"),
    "next in init": (_model("ASSIGN init(x) := next(x); next(x) := x;"),
                     r"`next\(\)` is not allowed here"),
    "undeclared": (_model("ASSIGN init(x) := TRUE; next(x) := y;"),
                   r"`y` is not declared \(in next\(x\)\)"),
    "case without TRUE": (
        _model("ASSIGN init(x) := TRUE;\n"
               "next(x) := case x : FALSE; !x : TRUE; esac;"),
        "the last branch of a `case` must have condition TRUE"),
    "INIT constraint": (_model("INIT x | !x\nASSIGN next(x) := x;"),
                        "line 3: INIT is not `x = e`"),
    "TRANS constraint": (_model("ASSIGN init(x) := TRUE;\nTRANS next(x) | x"),
                         r"TRANS is not `next\(x\) = e`"),
    "INVAR constraint": (_model("ASSIGN init(x) := TRUE; next(x) := x;\n"
                                "INVAR x | !x"),
                         "INVAR is not `x = e`"),
    "FAIRNESS": (_model("ASSIGN init(x) := TRUE; next(x) := x;\nFAIRNESS x;"),
                 "FAIRNESS constraints are not read"),
    "JUSTICE": (_model("ASSIGN init(x) := TRUE; next(x) := x;\nJUSTICE x;"),
                "JUSTICE constraints are not read"),
    "COMPASSION": (_model("ASSIGN init(x) := TRUE; next(x) := x;\n"
                          "COMPASSION (x, x);"),
                   "COMPASSION constraints are not read"),
    "assigned twice": (_model("ASSIGN init(x) := TRUE; init(x) := FALSE;\n"
                              "next(x) := x;"),
                       "`x` is assigned twice"),
    "input assigned": (_model("ASSIGN init(x) := TRUE; next(x) := x;\n"
                              "next(i) := x;",
                              "IVAR i : boolean; VAR x : boolean;"),
                       "input variable `i` is assigned"),
    "undeclared assigned": (_model("ASSIGN init(x) := TRUE; next(x) := x;\n"
                                   "next(y) := x;"),
                            "`y` is assigned but not declared"),
    "frozen next": (_model("ASSIGN init(x) := TRUE; next(x) := x;",
                           "FROZENVAR x : boolean;"),
                    "FROZENVAR `x` is assigned a next value"),
    "every state and init": (_model("ASSIGN x := TRUE; init(x) := FALSE;"),
                             "in every state and also"),
    "cycle": (_model("ASSIGN init(x) := TRUE; init(y) := TRUE;\n"
                     "next(x) := next(y); next(y) := next(x);",
                     "VAR x : boolean; y : boolean;"),
              "the next state depends on itself"),
    "nonlinear": (_model("ASSIGN init(x) := 1; next(x) := x * x;",
                         "VAR x : integer;"),
                  "`\\*` of two non-constants"),
    "division": (_model("ASSIGN init(x) := 1; next(x) := x / 2;",
                        "VAR x : integer;"),
                 "`/` is not linear integer arithmetic"),
    "mod": (_model("ASSIGN init(x) := 1; next(x) := x mod 2;",
                   "VAR x : integer;"),
            "`mod` is not linear integer arithmetic"),
    "boolean as integer": (_model("ASSIGN init(x) := 1; next(x) := x + b;\n"
                                  "init(b) := TRUE; next(b) := b;",
                                  "VAR x : integer; b : boolean;"),
                           "a boolean where a integer is needed"),
    "integer to boolean": (_model("ASSIGN init(b) := 0; next(b) := b;",
                                  "VAR b : boolean;"),
                           "a integer where a boolean is needed"),
    "bare number": (_model("ASSIGN init(w) := 0; next(w) := w;",
                           "VAR w : unsigned word[4];"),
                    "the number 0 is not a word constant"),
    "signed division": (_model("ASSIGN init(w) := 0sd4_1;\n"
                               "next(w) := w / 0sd4_1;",
                               "VAR w : signed word[4];"),
                        "`/` on signed words"),
    "signed mod": (_model("ASSIGN init(w) := 0sd4_1;\n"
                          "next(w) := w mod 0sd4_1;",
                          "VAR w : signed word[4];"),
                   "`mod` on signed words"),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_refused(case):
    text, match = REFUSED[case]
    with pytest.raises(SMVRefused, match=match):
        parse_smv(text, strict=True)


def test_hrm_is_refused_for_its_free_variables():
    text = (FIXTURES / "hrm_word_level_no_prop.smv").read_text()
    with pytest.raises(SMVRefused) as e:
        parse_smv(text, strict=True)
    assert "`hrm_enhanced_ebmc_dut_MAX_BPM` has no initial value" in str(e.value)
    assert "TRANS" not in str(e.value)


def test_a_model_without_words_is_lia():
    from zrth import Bool, Int

    text = (FIXTURES / "mod10.smv").read_text()
    module, names = parse_smv(text, strict=True)
    assert names["x"].dtype == Int([1, 1])
    assert names["inc"].dtype == Bool([1, 1])
    assert [str(v) for v in module.ctrl] == [str(names["x"])]
    assert [str(v) for v in module.extl] == [str(names["inc"])]
    trace = _run(text, [{"inc": (1, 0)}] * 10 + [{"inc": (0, 1)}])
    assert [s["x"] for s in trace] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0]


def test_mod10_has_a_lean_form():
    from zrth.lean.codegen.context import LeanContext
    from zrth.lean.codegen.terms import lean_gaps

    module, _ = parse_smv((FIXTURES / "mod10.smv").read_text(), strict=True)
    assert lean_gaps(LeanContext(module)) == []


def test_lia_operators():
    text = """MODULE main
VAR x : integer;
    n : integer;
    b : boolean;
    s : {idle, busy};
ASSIGN
  init(x) := -3;  next(x) := 2 * x + abs(n);
  init(n) := -1;  next(n) := n - 1;
  init(b) := FALSE; next(b) := (x < 0 -> b) <-> (n != -1);
  init(s) := idle; next(s) := s = idle ? busy : idle;
"""
    trace = _run(text, [{}, {}])
    assert [(t["x"], t["n"], t["b"], t["s"]) for t in trace] == [
        (-3, -1, 0, 0), (-5, -2, 1, 1), (-8, -3, 1, 0)]


@pytest.mark.parametrize("strict", [True, False])
def test_implies_and_iff_are_read_on_words(strict):
    text = """MODULE main
VAR a : boolean;
    b : boolean;
    w : unsigned word[1];
ASSIGN
  init(a) := FALSE; next(a) := !a;
  init(b) := TRUE;  next(b) := (a -> b) <-> a;
  init(w) := 0ud1_0; next(w) := w;
"""
    trace = _run(text, [{}, {}], strict=strict)
    # a: 0 1 0; b: 1, ((0->1)<->0)=0, ((1->0)<->1)=0
    assert [(t["a"], t["b"]) for t in trace] == [(0, 1), (1, 0), (0, 0)]
