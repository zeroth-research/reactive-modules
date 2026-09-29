import Zrth.Theory.LIA
import Zrth.DataFlow.Diagram
import Zrth.Reactive

/-!
# Two-bit counter

A two-bit counter `b1 b0` over LIA that increments when `enable` is set and
holds otherwise (the `twobitcounter` of `python/tests/test_eval.py`):

```
init:    b0' := false
         b1' := false
update:  b0' := if enable' then ¬b0 else b0
         b1' := if b0 ∧ enable' then ¬b1 else b1
```
-/

namespace Zrth.Examples.TwoBit

/-! ## Variables -/

abbrev VarLIA := Var SortsLIA

def b0 : VarLIA := { name := 0, sort := .bool 1 1 }
def b1 : VarLIA := { name := 1, sort := .bool 1 1 }
def enable : VarLIA := { name := 2, sort := .bool 1 1 }

/-! ## Reactive module -/

/-- The only atom: controls both bits, awaits `enable`. -/
def atom : Atom SortsLIA where
  read := [b0, b1]
  wait := [enable]
  ctrl := [b0, b1]
  ctrl_f := []

  init := { dom := [.bool 1 1], cod := [.bool 1 1, .bool 1 1] }
  update := { dom := [.bool 1 1, .bool 1 1, .bool 1 1], cod := [.bool 1 1, .bool 1 1] }
  flow := { dom := [.bool 1 1, .bool 1 1, .zero], cod := [] }

/-- The counter: `enable` comes from the environment, the bits are visible. -/
def counter : Module SortsLIA where
  extl := [enable]
  intf := [b0, b1]
  prvt := []
  atoms := [atom]

/-! ## Data-flow diagrams of the atom -/

/-- Constant `false`. -/
def ff : Mat Bool 1 1 := fun _ _ => false

-- wires of the latched (`b0`, `b1`) and the current-round (`*'`) values
def wB0 : Wire SortsLIA := ⟨0, .bool 1 1⟩
def wB1 : Wire SortsLIA := ⟨1, .bool 1 1⟩
def wEnable : Wire SortsLIA := ⟨2, .bool 1 1⟩
def wB0' : Wire SortsLIA := ⟨3, .bool 1 1⟩
def wB1' : Wire SortsLIA := ⟨4, .bool 1 1⟩
-- internal wires
def wNotB0 : Wire SortsLIA := ⟨5, .bool 1 1⟩
def wNotB1 : Wire SortsLIA := ⟨6, .bool 1 1⟩
def wB0AndEnable : Wire SortsLIA := ⟨7, .bool 1 1⟩

/-- `b0' := false; b1' := false` -/
def initDiagram : Diagram thrLIA where
  boxes := [
    { gen := .bool 1 1 ff, read := [], write := [wB0'] },
    { gen := .bool 1 1 ff, read := [], write := [wB1'] }
  ]
  read := [wEnable]
  write := [wB0', wB1']
  wires := [wEnable, wB0', wB1']

/-- `b0' := ite enable' (¬b0) b0; b1' := ite (b0 ∧ enable') (¬b1) b1` -/
def updateDiagram : Diagram thrLIA where
  boxes := [
    { gen := .not 1 1, read := [wB0], write := [wNotB0] },
    { gen := .ite (.bool 1 1), read := [wEnable, wNotB0, wB0], write := [wB0'] },
    { gen := .and 1 1, read := [wB0, wEnable], write := [wB0AndEnable] },
    { gen := .not 1 1, read := [wB1], write := [wNotB1] },
    { gen := .ite (.bool 1 1), read := [wB0AndEnable, wNotB1, wB1], write := [wB1'] }
  ]
  read := [wB0, wB1, wEnable]
  write := [wB0', wB1']
  wires := [wB0, wB1, wEnable, wB0', wB1', wNotB0, wNotB1, wB0AndEnable]

-- the diagrams fit the signatures of the atom
example : initDiagram.read.map Wire.sort = atom.init.dom := rfl
example : initDiagram.write.map Wire.sort = atom.init.cod := rfl
example : updateDiagram.read.map Wire.sort = atom.update.dom := rfl
example : updateDiagram.write.map Wire.sort = atom.update.cod := rfl

end Zrth.Examples.TwoBit
