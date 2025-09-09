use std::collections::{HashMap, HashSet};

/// This is an initial datastructure to store the state of wires/variables.
/// We currently fix identifiers to `String` and values to `usize` but that is going to change
/// in the future.
struct Store {
    state: HashMap<String, usize>,
}

/// A wiring corresponds to a set of identifiers.
/// We use it to denote a set of inputs or outputs to an operator/module.
#[derive(Clone)]
struct Wiring {
    wires: HashSet<String>,
}

impl Wiring {
    /// Compute the union between two wirings.
    fn union(&self, other: &Wiring) -> Wiring {
        Wiring {
            wires: self.wires.union(&other.wires).cloned().collect(),
        }
    }

    /// Compute the common wires between two wirings.
    fn intersection(&self, other: &Wiring) -> Wiring {
        Wiring {
            wires: self.wires.intersection(&other.wires).cloned().collect(),
        }
    }
}

/// An operator corresponds to:
/// - a box in a string diagram
/// - an atom in the original reactive modules description
/// In contrast to the original reactive modules, we do not explicitly declare wait variables here.
/// We only distinguish `read` and `write`. Wait variables are implicitly defined inside a [Module].
trait Operator {
    /// The input wiring to an operator.
    fn read(&self) -> &Wiring;
    /// The output wiring of an operator.
    fn write(&self) -> &Wiring;
}

/// Sequential operators can be interpreted and operate in *discrete time steps* as opposed to
/// [Differential], which operates in *continuous time steps*.
/// This corresponds to the Update action in reactive modules.
trait Sequential: Operator {
    /// Run a single step of the operator.
    /// `transition` contains both the current and next state as well as external parameters.
    fn update(&self, transition: &mut Store);
}

/// Operators can optionally initialize the state.
/// This corresponds to the Init action in reactive modules.
trait Initialized: Operator {
    /// Initializes the state. This potentially has access to external parameters
    /// and can write the initialized output wires.
    fn init(&self, store: &mut Store);
}

/// Differential operators can be interpreted and operate in *continuous time steps* as opposed to
/// [Sequential], which operates in *discrete time steps*.
/// This is a continuous time Update action.
trait Differential: Operator {
    /// Simulates passing of `time` for the operator.
    /// `transition` contains both the current and next state as well as external parameters.
    fn delay(&self, time: f64, transition: &mut Store);
}

/// A module corresponds to the concept of a reactive module.
/// It has access to the current (latched) and next states, which are represented through
/// distinct wirings in this design.
/// It also has a topologically ordered list of operators.  
///
/// Modules are also [Operator]s themselves, which allows fixed-depth nested structures.
struct Module<O: Operator> {
    current: Wiring,
    next: Wiring,
    operators: Vec<O>,
}

/// A generic runtime error for initializing a module (e.g. when no topological order exists).
/// TODO: needs better error handling
struct ModuleError;
impl<O: Operator> Module<O> {
    /// This is the user-facing default constructor for a module.
    /// It checks that wirings are matching and tries to determine a valid ordering.
    /// It fails if any of the checks fail.
    fn new(current: Wiring, next: Wiring, operators: Vec<O>) -> Result<Module<O>, ModuleError> {
        // whenever read of an operator is in `next` -> await
        todo!()
    }

    /// This is an unchecked variant of the constructor that might produce invalid modules.
    /// It only contains `debug_assert`s.
    fn new_unchecked(
        current_ext: Wiring,
        current_interface: Wiring,
        next: Wiring,
        operators: Vec<O>,
    ) -> Module<O> {
        // has debug assertions
        todo!()
    }
}

impl<O: Operator> Operator for Module<O> {
    fn read(&self) -> &Wiring {
        &self.current
    }

    fn write(&self) -> &Wiring {
        &self.next
    }
}

impl<O: Operator + Initialized> Initialized for Module<O> {
    fn init(&self, store: &mut Store) {
        for op in self.operators.iter() {
            op.init(store);
        }
    }
}

impl<O: Operator + Sequential> Sequential for Module<O> {
    fn update(&self, store: &mut Store) {
        for op in self.operators.iter() {
            op.update(store);
        }
    }
}

impl<O: Operator + Differential> Differential for Module<O> {
    fn delay(&self, time: f64, store: &mut Store) {
        for op in self.operators.iter() {
            op.delay(time, store);
        }
    }
}
