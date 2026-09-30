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

-- abbreviation for the LIA multi-sort
abbrev 𝕊 := SortsLIA
-- abbreviation for the 1×1 boolean matrix
abbrev 𝔹 := SortsLIA.bool 1 1

def b0 : Var 𝕊 := { name := 0, sort := 𝔹 }
def b1 : Var 𝕊  := { name := 1, sort := 𝔹 }
def enable : Var 𝕊 := { name := 2, sort := 𝔹 }

/-! ## Reactive module -/

/-- The only atom: controls both bits, awaits `enable`. -/
def atom : Atom 𝕊 where
  read := [b0, b1]
  wait := [enable]
  ctrl := [b0, b1]
  ctrl_f := []

  init := { dom := [𝔹], cod := [𝔹, 𝔹] }
  update := { dom := [𝔹, 𝔹, 𝔹], cod := [𝔹, 𝔹] }
  flow := { dom := [𝔹, 𝔹, .zero], cod := [] }

/-- The counter: `enable` comes from the environment, the bits are visible. -/
def counter : Module 𝕊 where
  extl := [enable]
  intf := [b0, b1]
  prvt := []
  atoms := [atom]



/- ------------------------------------------------------------ -/
/-! ## Data-flow diagrams of the atom -/
/- ------------------------------------------------------------ -/

/-- Constant `false`. -/
def ff : Mat Bool 1 1 := fun _ _ => false

-- wires of the latched (`b0`, `b1`) and the current-round (`*'`) values
def wB0 : Wire 𝕊  := ⟨0, 𝔹⟩
def wB1 : Wire 𝕊  := ⟨1, 𝔹⟩
def wEnable : Wire 𝕊 := ⟨2, 𝔹⟩
def wB0' : Wire 𝕊 := ⟨3, 𝔹⟩
def wB1' : Wire 𝕊 := ⟨4, 𝔹⟩
-- internal wires
def wNotB0 : Wire 𝕊  := ⟨5, 𝔹 ⟩
def wNotB1 : Wire 𝕊  := ⟨6, 𝔹 ⟩
def wB0AndEnable : Wire 𝕊 := ⟨7, 𝔹 ⟩

/-- `b0' := false; b1' := false` -/
def initDiagram : Diagram thrLIA where
  boxes := [
    { gen := .bool ⟨1, 1, ff⟩, read := [], write := [wB0'] },
    { gen := .bool ⟨1, 1, ff⟩, read := [], write := [wB1'] }
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
