import Zrth.Theory.LRA
import Zrth.DataFlow.Diagram
import Zrth.Reactive

/-!
# Clocks

A closed module over LRA with two real clocks, each controlled by its own
atom. `x` flows at rate `1` and `y` at rate `2`; an update resets a clock that
reached `5` to `0`:

```
init:    x' := 0
update:  x' := if x = 5 then 0 else x
flow:    ẋ  := 1

init:    y' := 0
update:  y' := if y = 5 then 0 else y
flow:    ẏ  := 2
```
-/

namespace Zrth.Examples.Clocks

/-! ## Variables -/

-- we do not import MathLib, so we can use ℝ here
abbrev ℝ := SortsLRA.real 1 1
abbrev δℝ := SortsLRA.δreal 1 1 0
abbrev 𝔹 := SortsLRA.bool 1 1

def x : Var SortsLRA := { name := 0, sort := ℝ }
def y : Var SortsLRA := { name := 1, sort := ℝ }

/-! ## Reactive module -/

/-- Controls the clock `x` and its flow. -/
def atomX : Atom SortsLRA where
  read := [x]
  wait := []
  ctrl := [x]
  ctrl_f := [x]

  init := { dom := [], cod := [ℝ] }
  update := { dom := [ℝ], cod := [ℝ] }
  flow := { dom := [ℝ], cod := [δℝ] }

/-- Controls the clock `y` and its flow. -/
def atomY : Atom SortsLRA where
  read := [y]
  wait := []
  ctrl := [y]
  ctrl_f := [y]

  init := { dom := [], cod := [ℝ] }
  update := { dom := [ℝ], cod := [ℝ] }
  flow := { dom := [ℝ], cod := [δℝ] }

/-- The clocks have no inputs; both clocks are visible. -/
def clocks : Module SortsLRA where
  extl := []
  intf := [x, y]
  prvt := []
  atoms := [atomX, atomY]

/- ------------------------------------------------------------ -/
/-! ## Data-flow diagrams of the atoms -/
/- ------------------------------------------------------------ -/

/-- The real constant `c`. -/
def k (c : Float) : Tensor Float := ⟨1, 1, fun _ _ => c⟩

-- constants shared by both atoms
def w0 : Wire SortsLRA := ⟨0, ℝ⟩
def w5 : Wire SortsLRA := ⟨1, ℝ⟩
def wZero : Wire SortsLRA := ⟨2, δℝ⟩

/-- Literals are values, never derivatives: the rate `c` is the affine map
`0 ↦ 1 · 0 + c` applied to the zero derivative. -/
def rate (c : Float) : LRA.Gen := .linear (k 1) (some (k c)) 1 1 (by simp [LRA.linearFits, k])

-- wires of `x`: the latched value, the next value, and the derivative
def wX : Wire SortsLRA := ⟨3, ℝ⟩
def wX' : Wire SortsLRA := ⟨4, ℝ⟩
def wDX : Wire SortsLRA := ⟨5, δℝ⟩
def wXIsFive : Wire SortsLRA := ⟨6, 𝔹⟩

/-- `x' := 0` -/
def initX : Diagram thrLRA where
  boxes := [{ gen := .real (k 0), read := [], write := [wX'] }]
  read := []
  write := [wX']
  wires := [wX']

/-- `x' := ite (x = 5) 0 x` -/
def updateX : Diagram thrLRA where
  boxes := [
    { gen := .real (k 0), read := [], write := [w0] },
    { gen := .real (k 5), read := [], write := [w5] },
    { gen := .eq 1 1, read := [wX, w5], write := [wXIsFive] },
    { gen := .ite ℝ, read := [wXIsFive, w0, wX], write := [wX'] }
  ]
  read := [wX]
  write := [wX']
  wires := [wX, wX', w0, w5, wXIsFive]

/-- `ẋ := 1` -/
def flowX : Diagram thrLRA where
  boxes := [
    { gen := .realZerograd 1 1, read := [], write := [wZero] },
    { gen := rate 1, read := [wZero], write := [wDX] }
  ]
  read := [wX]
  write := [wDX]
  wires := [wX, wDX, wZero]

-- wires of `y`: the latched value, the next value, and the derivative
def wY : Wire SortsLRA := ⟨7, ℝ⟩
def wY' : Wire SortsLRA := ⟨8, ℝ⟩
def wDY : Wire SortsLRA := ⟨9, δℝ⟩
def wYIsFive : Wire SortsLRA := ⟨10, 𝔹⟩

/-- `y' := 0` -/
def initY : Diagram thrLRA where
  boxes := [{ gen := .real (k 0), read := [], write := [wY'] }]
  read := []
  write := [wY']
  wires := [wY']

/-- `y' := ite (y = 5) 0 y` -/
def updateY : Diagram thrLRA where
  boxes := [
    { gen := .real (k 0), read := [], write := [w0] },
    { gen := .real (k 5), read := [], write := [w5] },
    { gen := .eq 1 1, read := [wY, w5], write := [wYIsFive] },
    { gen := .ite ℝ, read := [wYIsFive, w0, wY], write := [wY'] }
  ]
  read := [wY]
  write := [wY']
  wires := [wY, wY', w0, w5, wYIsFive]

/-- `ẏ := 2` -/
def flowY : Diagram thrLRA where
  boxes := [
    { gen := .realZerograd 1 1, read := [], write := [wZero] },
    { gen := rate 2, read := [wZero], write := [wDY] }
  ]
  read := [wY]
  write := [wDY]
  wires := [wY, wDY, wZero]

-- the diagrams fit the signatures of the atoms
example : initX.read.map Wire.sort = atomX.init.dom := rfl
example : initX.write.map Wire.sort = atomX.init.cod := rfl
example : updateX.read.map Wire.sort = atomX.update.dom := rfl
example : updateX.write.map Wire.sort = atomX.update.cod := rfl
example : flowX.read.map Wire.sort = atomX.flow.dom := rfl
example : flowX.write.map Wire.sort = atomX.flow.cod := rfl
example : initY.read.map Wire.sort = atomY.init.dom := rfl
example : initY.write.map Wire.sort = atomY.init.cod := rfl
example : updateY.read.map Wire.sort = atomY.update.dom := rfl
example : updateY.write.map Wire.sort = atomY.update.cod := rfl
example : flowY.read.map Wire.sort = atomY.flow.dom := rfl
example : flowY.write.map Wire.sort = atomY.flow.cod := rfl

end Zrth.Examples.Clocks
