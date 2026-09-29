import Zrth.Theory.LIA
import Zrth.DataFlow.Diagram
import Zrth.Reactive

/-!
# Toggle

A closed module over LIA with a single bit that flips every round:

```
init:    t' := false
update:  t' := ¬t
```
-/

namespace Zrth.Examples.Toggle

/-! ## Variables -/

def t : @Var SortsLIA := { name := 0, sort := .bool 1 1 }

/-! ## Reactive module -/

def atom : @Atom SortsLIA _ where
  read := [t]
  wait := []
  ctrl := [t]
  ctrl_f := []

  init := { dom := [], cod := [.bool 1 1] }
  update := { dom := [.bool 1 1], cod := [.bool 1 1] }
  flow := { dom := [.bool 1 1], cod := [] }

/-- The toggle has no inputs; the bit is visible. -/
def toggle : @Module SortsLIA _ where
  extl := []
  intf := [t]
  prvt := []
  atoms := [atom]

/-! ## Data-flow diagrams of the atom -/

def wT : Wire SortsLIA := ⟨0, .bool 1 1⟩
def wT' : Wire SortsLIA := ⟨1, .bool 1 1⟩

/-- `t' := false` -/
def initDiagram : Diagram thrLIA where
  boxes := [{ gen := .bool 1 1 (fun _ _ => false), read := [], write := [wT'] }]
  read := []
  write := [wT']
  wires := [wT']

/-- `t' := ¬t` -/
def updateDiagram : Diagram thrLIA where
  boxes := [{ gen := .not 1 1, read := [wT], write := [wT'] }]
  read := [wT]
  write := [wT']
  wires := [wT, wT']

-- the diagrams fit the signatures of the atom
example : initDiagram.read.map Wire.sort = atom.init.dom := rfl
example : initDiagram.write.map Wire.sort = atom.init.cod := rfl
example : updateDiagram.read.map Wire.sort = atom.update.dom := rfl
example : updateDiagram.write.map Wire.sort = atom.update.cod := rfl

end Zrth.Examples.Toggle
