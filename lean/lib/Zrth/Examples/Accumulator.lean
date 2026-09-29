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

def x : Var SortsLRA := { name := 0, sort := .real 1 1 }
def u : Var SortsLRA := { name := 1, sort := .real 1 1 }

/-! ## Reactive module -/

/-- The only atom: controls `x`, awaits `u`. -/
def atom : Atom SortsLRA where
  read := [x]
  wait := [u]
  ctrl := [x]
  ctrl_f := []

  init := { dom := [.real 1 1], cod := [.real 1 1] }
  update := { dom := [.real 1 1, .real 1 1], cod := [.real 1 1] }
  flow := { dom := [.real 1 1, .real 1 1 1], cod := [] }

/-- The accumulator: `u` comes from the environment, the sum is visible. -/
def accumulator : Module SortsLRA where
  extl := [u]
  intf := [x]
  prvt := []
  atoms := [atom]

/-! ## Data-flow diagrams of the atom -/

def wX : Wire SortsLRA := ⟨0, .real 1 1⟩
def wU : Wire SortsLRA := ⟨1, .real 1 1⟩
def wX' : Wire SortsLRA := ⟨2, .real 1 1⟩

/-- `x' := u'` -/
def initDiagram : Diagram thrLRA where
  boxes := [{ gen := .id (.real 1 1), read := [wU], write := [wX'] }]
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
