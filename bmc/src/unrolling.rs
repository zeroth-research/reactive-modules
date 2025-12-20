use std::collections::HashMap;

use base::{Interface, Term, Wire};
use common::context::Context;
use common::transition::{Transition, WiredTransitions};

/// A helper struct to keep mapping between wire IDs
struct WiresMapping {
    map: HashMap<usize, usize>,
}

impl WiresMapping {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Add new old_wire_id -> new_wire_id mappings.
    /// old_wire_id must not be already present in the mapping
    pub fn add_mapping<T1, T2>(&mut self, old: T1, new: T2)
    where
        T1: IntoIterator<Item = usize>,
        T2: IntoIterator<Item = usize>,
    {
        // do the zipping manually so that we can check the lengths too
        let mut old = old.into_iter();
        let mut new = new.into_iter();
        loop {
            match (old.next(), new.next()) {
                (Some(id_old), Some(id_new)) => {
                    self.insert(id_old, id_new);
                }
                (None, None) => return,
                _ => panic!("Old and new IDs have different lengths"),
            }
        }
    }

    /// Get an existing mapping
    pub fn get(&mut self, id: usize) -> Option<usize> {
        self.map.get(&id).copied()
    }

    /// Insert new mapping
    pub fn insert(&mut self, id: usize, new_id: usize) {
        if self.map.insert(id, new_id).is_some() {
            panic!("Already have mapping for this ID")
        }
    }
}

/// Unrolling of a module
///
/// This struct just wraps the *process* of unrolling. Its methods take
/// a sequence of transitions in the form of [WiredTransitions] object
/// and initialize/extend it with terms from the module.
/// The sequence of transitions is external to [ModuleUnrolling].
pub struct ModuleUnrolling<'c, 'm, D: Copy + Eq, I> {
    pub(crate) ctx: &'c mut Context<D>,
    pub(crate) module: &'m base::Module<D, I>,
}

impl<'c, 'm, D: Copy + Eq + std::fmt::Debug, I: Clone + std::fmt::Debug>
    ModuleUnrolling<'c, 'm, D, I>
{
    pub fn new(module: &'m base::Module<D, I>, ctx: &'c mut Context<D>) -> Self {
        ModuleUnrolling::<'c, 'm, D, I> { ctx, module }
    }

    // Initialize the sequence of transitions from the module init.
    pub fn init(&mut self, mut transitions: WiredTransitions<D, I>) -> WiredTransitions<D, I> {
        self.init_ref(&mut transitions);
        transitions
    }

    // Extend a sequence of transitions with terms from the `update` of the module
    pub fn step(&mut self, mut unrolling: WiredTransitions<D, I>) -> WiredTransitions<D, I> {
        self.step_ref(&mut unrolling);
        unrolling
    }

    // Initialize the sequence of transitions from the module init.
    pub fn init_ref(&mut self, transitions: &mut WiredTransitions<D, I>) {
        assert!(transitions.is_empty());

        // create the initial transition
        let transition = Transition::from_module_init(self.module).unwrap();

        transitions.push(transition).unwrap();
    }

    // Extend a sequence of transitions with terms from the `update` of the module
    pub fn step_ref(&mut self, transitions: &mut WiredTransitions<D, I>) {
        assert!(!transitions.is_empty());

        // the input interface here is the last interface in this transitions
        let intf_in =
            Interface::sequence(transitions.last_interface().unwrap().wires().cloned()).unwrap();

        // environment interface is a fresh "copy" of the environment interface
        let intf_env = if !self.module.extl().is_empty() {
            Some(
                Interface::try_from_iter(self.module.extl().iter().map(|w| {
                    debug_assert!(w[0].dtype() == w[1].dtype());
                    [
                        self.ctx.tmp_wire(*w[0].dtype()),
                        self.ctx.tmp_wire(*w[1].dtype()),
                    ]
                }))
                .unwrap(),
            )
        } else {
            None
        };

        // the output interface is a fresh "copy" of control variables
        let intf_out = self.ctx.fresh_intf(
            // TODO: do not create a new interface here, make `fresh_intf` take an iterator
            &intf_in.clone(),
        );

        // create mapping between interface wires
        let mut wire_map = WiresMapping::new();

        // map input of the transition to output of the sequence (which is the current `intf_in`)
        wire_map.add_mapping(
            self.module.ctrl().latched().iter().map(Wire::id),
            intf_in.ids(),
        );
        // map output of the transition to new output ids
        wire_map.add_mapping(
            self.module.ctrl().next().iter().map(Wire::id),
            intf_out.ids(),
        );
        // map externals to the new env
        if let Some(ienv) = &intf_env {
            wire_map.add_mapping(
                self.module.extl().iter().map(|i| i[0].id()),
                ienv.iter().map(|i| i[0].id()),
            );
            wire_map.add_mapping(
                self.module.extl().iter().map(|i| i[1].id()),
                ienv.iter().map(|i| i[1].id()),
            );
        }

        // take the update terms and re-map them to use new wires
        let mut terms: Vec<Term<D, I>> = Vec::new();
        for atom in self.module.atoms() {
            for term in atom.update() {
                terms.push(self.map_term(term, &mut wire_map));
            }
        }

        // create the update transition
        let transition = Transition::new(intf_in, intf_env, intf_out, terms);

        transitions.push(transition).unwrap();
    }

    /// Re-map term to use new IDs
    fn map_term(&mut self, term: &Term<D, I>, map: &mut WiresMapping) -> Term<D, I> {
        // remap read wires
        let read: Vec<Wire<D>> = term
            .read()
            .iter()
            .map(|[w]| {
                let new_id = if let Some(wid) = map.get(w.id()) {
                    wid
                } else {
                    let new = self.ctx.tmp_id();
                    map.insert(w.id(), new);
                    new
                };
                Wire::new(new_id, *w.dtype())
            })
            .collect();

        // remap write wires
        let write: Vec<Wire<D>> = term
            .write()
            .iter()
            .map(|[w]| {
                let new_id = if let Some(wid) = map.get(w.id()) {
                    wid
                } else {
                    let new = self.ctx.tmp_id();
                    map.insert(w.id(), new);
                    new
                };
                Wire::new(new_id, *w.dtype())
            })
            .collect();

        Term::function(term.itype().clone(), write, read).unwrap()
    }
}
