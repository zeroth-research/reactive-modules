import Zrth.Theory.LIA
import Zrth.DataFlow.Diagram
import Zrth.Reactive

/-!
# Counter

Two counters over LIA that count from `0` to `10` and then reset to `0`,
each updated by its own atom. When the environment sets `hold` while `c` is at
`1`, `c` stays at `1`; `d` counts unconditionally:

```
init:    c' := 0
update:  c' := if hold' ∧ c = 1 then c
               else if c = 10 then 0
               else c + 1

init:    d' := 0
update:  d' := if d = 10 then 0 else d + 1
```
-/

namespace Zrth.Examples.Counter

/-! ## Variables -/

-- abbreviation for the LIA multi-sort
abbrev 𝕊 := SortsLIA
-- abbreviations for the 1×1 integer and boolean matrices
abbrev ℤ := SortsLIA.int 1 1
abbrev 𝔹 := SortsLIA.bool 1 1

def c : Var 𝕊 := { name := 0, sort := ℤ }
def hold : Var 𝕊 := { name := 1, sort := 𝔹 }
def d : Var 𝕊 := { name := 2, sort := ℤ }

/-! ## Reactive module -/

/-- Controls `c`, awaits `hold`. -/
def atomC : Atom 𝕊 where
  read := [c]
  wait := [hold]
  ctrl := [c]
  ctrl_f := []

  init := { dom := [𝔹], cod := [ℤ] }
  update := { dom := [ℤ, 𝔹], cod := [ℤ] }
  flow := { dom := [ℤ, .zero], cod := [] }

/-- Controls `d`, awaits nothing. -/
def atomD : Atom 𝕊 where
  read := [d]
  wait := []
  ctrl := [d]
  ctrl_f := []

  init := { dom := [], cod := [ℤ] }
  update := { dom := [ℤ], cod := [ℤ] }
  flow := { dom := [ℤ], cod := [] }

/-- The counters: `hold` comes from the environment, `c` and `d` are visible. -/
def counter : Module 𝕊 where
  extl := [hold]
  intf := [c, d]
  prvt := []
  atoms := [atomC, atomD]

/- ------------------------------------------------------------ -/
/-! ## Data-flow diagrams of the atoms -/
/- ------------------------------------------------------------ -/

/-- The integer constant `x`. -/
def k (x : Int) : Tensor Int := ⟨1, 1, fun _ _ => x⟩

-- wires of the latched (`c`) and the current-round (`*'`) values
def wC : Wire 𝕊 := ⟨0, ℤ⟩
def wHold : Wire 𝕊 := ⟨1, 𝔹⟩
def wC' : Wire 𝕊 := ⟨2, ℤ⟩
-- internal wires
def w0 : Wire 𝕊 := ⟨3, ℤ⟩
def w1 : Wire 𝕊 := ⟨4, ℤ⟩
def w10 : Wire 𝕊 := ⟨5, ℤ⟩
def wIsOne : Wire 𝕊 := ⟨6, 𝔹⟩
def wStay : Wire 𝕊 := ⟨7, 𝔹⟩
def wIsTen : Wire 𝕊 := ⟨8, 𝔹⟩
def wSucc : Wire 𝕊 := ⟨9, ℤ⟩
def wStep : Wire 𝕊 := ⟨10, ℤ⟩

/-- `c' := 0` -/
def initC : Diagram thrLIA where
  boxes := [{ gen := .int (k 0), read := [], write := [wC'] }]
  read := [wHold]
  write := [wC']
  wires := [wHold, wC']

/-- `c' := ite (hold' ∧ c = 1) c (ite (c = 10) 0 (c + 1))` -/
def updateC : Diagram thrLIA where
  boxes := [
    { gen := .int (k 0), read := [], write := [w0] },
    { gen := .int (k 1), read := [], write := [w1] },
    { gen := .int (k 10), read := [], write := [w10] },
    -- stay at `1` when the environment holds
    { gen := .eq 1 1, read := [wC, w1], write := [wIsOne] },
    { gen := .and 1 1, read := [wHold, wIsOne], write := [wStay] },
    -- otherwise count up to `10`, then reset
    { gen := .eq 1 1, read := [wC, w10], write := [wIsTen] },
    { gen := .add 1 1, read := [wC, w1], write := [wSucc] },
    { gen := .ite ℤ, read := [wIsTen, w0, wSucc], write := [wStep] },
    { gen := .ite ℤ, read := [wStay, wC, wStep], write := [wC'] }
  ]
  read := [wC, wHold]
  write := [wC']
  wires := [wC, wHold, wC', w0, w1, w10, wIsOne, wStay, wIsTen, wSucc, wStep]

-- wires of `d` (the constants are shared with `c`)
def wD : Wire 𝕊 := ⟨11, ℤ⟩
def wD' : Wire 𝕊 := ⟨12, ℤ⟩
def wDIsTen : Wire 𝕊 := ⟨13, 𝔹⟩
def wDSucc : Wire 𝕊 := ⟨14, ℤ⟩

/-- `d' := 0` -/
def initD : Diagram thrLIA where
  boxes := [{ gen := .int (k 0), read := [], write := [wD'] }]
  read := []
  write := [wD']
  wires := [wD']

/-- `d' := ite (d = 10) 0 (d + 1)` -/
def updateD : Diagram thrLIA where
  boxes := [
    { gen := .int (k 0), read := [], write := [w0] },
    { gen := .int (k 1), read := [], write := [w1] },
    { gen := .int (k 10), read := [], write := [w10] },
    { gen := .eq 1 1, read := [wD, w10], write := [wDIsTen] },
    { gen := .add 1 1, read := [wD, w1], write := [wDSucc] },
    { gen := .ite ℤ, read := [wDIsTen, w0, wDSucc], write := [wD'] }
  ]
  read := [wD]
  write := [wD']
  wires := [wD, wD', w0, w1, w10, wDIsTen, wDSucc]

-- the diagrams fit the signatures of the atoms
example : initC.read.map Wire.sort = atomC.init.dom := rfl
example : initC.write.map Wire.sort = atomC.init.cod := rfl
example : updateC.read.map Wire.sort = atomC.update.dom := rfl
example : updateC.write.map Wire.sort = atomC.update.cod := rfl
example : initD.read.map Wire.sort = atomD.init.dom := rfl
example : initD.write.map Wire.sort = atomD.init.cod := rfl
example : updateD.read.map Wire.sort = atomD.update.dom := rfl
example : updateD.write.map Wire.sort = atomD.update.cod := rfl

end Zrth.Examples.Counter
