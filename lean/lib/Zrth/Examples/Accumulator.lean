import Zrth.Theory.LRA
import Zrth.DataFlow.Diagram
import Zrth.Reactive

/-!
# Accumulator

A module over LRA summing its input `u`, starting from the first input:

```
init:    x' := u'
update:  x' := x + u'
```
-/

namespace Zrth.Examples.Accumulator

/-! ## Variables -/

-- we do not import MathLib, so we can use ℝ here
abbrev ℝ := SortsLRA.real 1 1
abbrev δℝ := SortsLRA.δreal 1 1 0

def x : Var SortsLRA := { name := 0, sort := ℝ }
def u : Var SortsLRA := { name := 1, sort := ℝ }

/-! ## Reactive module -/

/-- The only atom: controls `x`, awaits `u`. -/
def atom : Atom SortsLRA where
  read := [x]
  wait := [u]
  ctrl := [x]
  ctrl_f := []

  init := { dom := [ℝ], cod := [ℝ] }
  update := { dom := [ℝ, ℝ], cod := [ℝ] }
  flow := { dom := [ℝ, δℝ], cod := [] }

/-- The accumulator: `u` comes from the environment, the sum is visible. -/
def accumulator : Module SortsLRA where
  extl := [u]
  intf := [x]
  prvt := []
  atoms := [atom]

/-! ## Data-flow diagrams of the atom -/

def wX : Wire SortsLRA := ⟨0, ℝ⟩
def wU : Wire SortsLRA := ⟨1, ℝ⟩
def wX' : Wire SortsLRA := ⟨2, ℝ⟩

/-- `x' := u'` -/
def initDiagram : Diagram thrLRA where
  boxes := [{ gen := .id (ℝ), read := [wU], write := [wX'] }]
  read := [wU]
  write := [wX']
  wires := [wU, wX']

/-- `x' := x + u'` -/
def updateDiagram : Diagram thrLRA where
  boxes := [{ gen := .add 1 1, read := [wX, wU], write := [wX'] }]
  read := [wX, wU]
  write := [wX']
  wires := [wX, wU, wX']

-- the diagrams fit the signatures of the atom
example : initDiagram.read.map Wire.sort = atom.init.dom := rfl
example : initDiagram.write.map Wire.sort = atom.init.cod := rfl
example : updateDiagram.read.map Wire.sort = atom.update.dom := rfl
example : updateDiagram.write.map Wire.sort = atom.update.cod := rfl

end Zrth.Examples.Accumulator
