use crate::context::Context;
use base::{Interface, Term, Wire};
use std::collections::{HashMap, HashSet};

type Err = &'static str;

/// Encoding of a transition relation between two (ordered) sets of variables.
///
/// [Transition] represents going form `intf_in` to `intf_out` (under environemt
/// inputs `intf_env` if given).
/// The transition itself is represented by the field `transition` which is
/// a sequence of [Term]s. The interface `intf_in` describes the input interface,
/// i.e., what wires in `transition` will get the input values. The interface `intf_out` are
/// the output wires. Field `intf_env` are optional inputs to `transition`, so they could
/// be a part of `intf_in`, but it is useful to keep them separately (for reasons, see [unrolling::Unrolling])
///
///
/// If the transition represents one round of a reactive module, then `intf_in = crtl[0]`,
/// `intf_env = extl[1]`, `intf_out = ctrl[1]` for an update round. For an init round,
/// it is the same except `intf_in` is empty. This is also a reason why we cannot have
/// `intf` of type `Interface<DType, 2>` instead of `intf_in` and `intf_out` (we couldn't
/// have `intf[0]` empty while `intf[1]` non-empty in the initial round).
/// We can change that in the future and have `InitTransition` and `UpdateTransition` or something
/// like that, if necessary.
///
///
/// # Examples
///
/// Consider symbolic transition systems. Assume you have a formula `transition` over variables
/// X and X' representing the transition relation (from current to primed, i.e., next, state variables),
/// and you have two sets of variables Y_1, and Y_2 (disjoint with X but with bijection to X).
/// Then `Transition(Y_1, None, Y_2, transition)` represents the formula `transition[X <- Y_1, X' <- Y]`
/// where `[X <- Y]` is the substitution of variables Y to X.
///
#[derive(Debug, Clone)]
pub struct Transition<D, I> {
    // input interface
    intf_in: Interface<D>,
    // output interface
    intf_out: Interface<D>,
    // external (environment) interface
    // it has old and current variables
    intf_env: Option<Interface<D, 2>>,

    // TODO: remove pub
    pub transition: Vec<Term<D, I>>,
}

impl<D, I> Transition<D, I> {
    pub fn new(
        intf_in: Interface<D>,
        intf_env: Option<Interface<D, 2>>,
        intf_out: Interface<D>,
        transition: Vec<Term<D, I>>,
    ) -> Self {
        // check that every input and output wires are used
        // (we do not require this for env. wires)
        #[cfg(debug_assertions)]
        {
            let mut ins: HashSet<usize> = HashSet::from_iter(intf_in.ids());
            let mut outs: HashSet<usize> = HashSet::from_iter(intf_out.ids());
            for term in &transition {
                if !ins.is_empty() {
                    for id in term.read().ids() {
                        ins.remove(&id);
                    }
                }

                if !outs.is_empty() {
                    for id in term.write().ids() {
                        outs.remove(&id);
                    }
                }
                if ins.is_empty() && outs.is_empty() {
                    break;
                }
            }
            debug_assert!(ins.is_empty(), "An input wire is not used by transition");
            debug_assert!(outs.is_empty(), "An ouput wire is not used by transition");
        }

        Self {
            intf_in,
            intf_env,
            intf_out,
            transition,
        }
    }

    pub fn intf_in(&self) -> &Interface<D> {
        &self.intf_in
    }

    pub fn intf_env(&self) -> Option<&Interface<D, 2>> {
        self.intf_env.as_ref()
    }

    pub fn intf_out(&self) -> &Interface<D> {
        &self.intf_out
    }

    pub fn terms(&self) -> std::slice::Iter<'_, Term<D, I>> {
        self.transition.iter()
    }
}

impl<D: Clone + Eq, I: Clone> Transition<D, I> {
    /// Create [Transition]s from [Module] wires and terms.
    /// The function returns a pair of [Transition]s, one for the init of the module
    /// and one for the update of the module
    pub fn from_module(module: &base::Module<D, I>) -> Result<(Self, Self), Err> {
        Ok((
            Self::from_module_init(module)?,
            Self::from_module_update(module)?,
        ))
    }

    /// Create [Transition]s from [Module] init wires and terms.
    pub fn from_module_init(module: &base::Module<D, I>) -> Result<Self, Err> {
        // the input interface here is empty
        let intf_in: Interface<D> = Interface::empty();

        // the environemnt inputs are the external wires of the module
        // (although the init can use only 'next' wires, we have to clone
        // both latched and next)
        let intf_env = module.extl().clone();

        // the output interface here are the *next* controlled wires
        let intf_out = Interface::sequence(module.ctrl().next().iter().cloned())?;

        // these are the initial terms, we copy them
        let mut terms: Vec<Term<D, I>> = Vec::new();
        for atom in module.atoms() {
            for term in atom.init() {
                terms.push(term.clone());
            }
        }

        Ok(Transition::new(intf_in, Some(intf_env), intf_out, terms))
    }

    /// Create [Transition]s from [Module] update wires and terms.
    pub fn from_module_update(module: &base::Module<D, I>) -> Result<Self, Err> {
        // the input interface are the latched control wires

        let intf_in = Interface::sequence(module.ctrl().latched().iter().cloned())?;

        // the output interface here are the *next* control wires
        let intf_out = Interface::sequence(module.ctrl().next().iter().cloned())?;

        // the environment are a copy of the external wires (if there are any)
        let intf_env = module.extl().clone();

        // clone the update terms
        let mut terms: Vec<Term<D, I>> = Vec::new();
        for atom in module.atoms() {
            for term in atom.update() {
                terms.push(term.clone());
            }
        }

        Ok(Transition::new(intf_in, Some(intf_env), intf_out, terms))
    }
}

impl<'a, D, I> IntoIterator for &'a Transition<D, I>
where
    D: 'a,
    I: 'a,
{
    type Item = &'a Term<D, I>;
    type IntoIter = std::slice::Iter<'a, Term<D, I>>;

    fn into_iter(self) -> Self::IntoIter {
        self.transition.iter()
    }
}

impl<D, I> IntoIterator for Transition<D, I> {
    type Item = Term<D, I>;
    type IntoIter = std::vec::IntoIter<Term<D, I>>;

    fn into_iter(self) -> Self::IntoIter {
        self.transition.into_iter()
    }
}

/// A wired sequence of transitions.
///
/// A sequence of [Transition]s where the output interface
/// of every transition wires with the input of the next transition.
/// Note that by the input interface here we mean `Transition::intf_in`.
/// The environment inputs `Transition::intf_env` may or may not be wired.
#[derive(Debug)]
pub struct WiredTransitions<D, I> {
    // TODO: remove pub
    pub transitions: Vec<Transition<D, I>>,
}

impl<D: Eq, I> Default for WiredTransitions<D, I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D, I> WiredTransitions<D, I> {
    pub fn new() -> Self {
        Self {
            transitions: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Transition<D, I>> {
        self.transitions.iter()
    }
}

impl<D: Eq, I> WiredTransitions<D, I> {
    /// Push a new transition to this sequence.
    ///
    /// If we're in a debug build, wiring is checked. Otherwise,
    /// this function always returns `Ok`.
    pub fn push(&mut self, t: Transition<D, I>) -> Result<(), String> {
        #[cfg(debug_assertions)]
        {
            if let Some(last) = self.transitions.last() {
                if last.intf_out().len() != t.intf_in().len() {
                    return Err(
                        "Transition does not wire to the sequence (different lenghts of interfaces)"
                            .into(),
                    );
                }
                for (n, (wo, wi)) in last.intf_out().wires().zip(t.intf_in().wires()).enumerate() {
                    if wi.id() != wo.id() {
                        return Err(format!(
                            "Input wire with idx {} does not wire to output wire (ID {} != ID {})",
                            n,
                            wo.id(),
                            wi.id(),
                        ));
                    }
                    if *wi.dtype() != *wo.dtype() {
                        return Err(format!(
                            "Input wire with idx {} does not have the same type as the output wire",
                            n
                        ));
                    }
                }
            }
        }

        self.transitions.push(t);
        Ok(())
    }

    pub fn last_interface(&self) -> Option<&Interface<D>> {
        Some(self.transitions.last()?.intf_out())
    }
}

mod aux {
    use super::*;

    /// A helper struct to keep mapping between wire IDs
    pub struct WiresMapping {
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

    /// A helper function to  re-map term to use new IDs
    pub fn map_term<D: Clone + Eq, I: Clone>(
        term: &Term<D, I>,
        map: &mut WiresMapping,
        ctx: &mut Context<D>,
    ) -> Result<Term<D, I>, Err> {
        // remap read wires
        let read: Vec<Wire<D>> = term
            .read()
            .iter()
            .map(|[w]| {
                let new_id = if let Some(wid) = map.get(w.id()) {
                    wid
                } else {
                    let new = ctx.tmp_id();
                    map.insert(w.id(), new);
                    new
                };
                Wire::new(new_id, w.dtype().clone())
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
                    let new = ctx.tmp_id();
                    map.insert(w.id(), new);
                    new
                };
                Wire::new(new_id, w.dtype().clone())
            })
            .collect();

        Term::function(term.itype().clone(), write, read)
    }
}

impl<D: Clone + Eq, I: Clone> WiredTransitions<D, I> {
    pub fn wire_transition(
        &mut self,
        t: &Transition<D, I>,
        ctx: &mut Context<D>,
    ) -> Result<(), Err> {
        if self.transitions.is_empty() {
            self.transitions.push(t.clone());
            return Ok(());
        }

        // the input interface of the new transition here is the last interface in this transitions
        let intf_in =
            Interface::sequence(self.transitions.last().unwrap().intf_out().wires().cloned())
                .unwrap();

        // the output interface is a fresh "copy" of input interface
        let intf_out = ctx.fresh_intf(
            // TODO: do not create a new interface here, make `fresh_intf` take an iterator
            &intf_in.clone(),
        );

        // environment interface is a fresh "copy" of the environment interface
        let intf_env = if let Some(intf_env) = &t.intf_env {
            Some(Interface::try_from_iter(intf_env.iter().map(|w| {
                debug_assert!(w[0].dtype() == w[1].dtype());
                [ctx.tmp_wire(w[0].dtype().clone()), ctx.tmp_wire(w[1].dtype().clone())]
            }))?)
        } else {
            None
        };

        //// create mapping between interface wires
        let mut wire_map = aux::WiresMapping::new();

        // map input of the transition to output of the sequence (which is the current `intf_in`)
        wire_map.add_mapping(t.intf_in().ids(), intf_in.ids());
        // map output of the transition to new output ids
        wire_map.add_mapping(t.intf_out().ids(), intf_out.ids());
        // map old env to the new env
        if let Some(ienv) = &t.intf_env {
            let ienv_new = intf_env.as_ref().unwrap();
            wire_map.add_mapping(
                ienv.iter().map(|i| i[0].id()),
                ienv_new.iter().map(|i| i[0].id()),
            );
            wire_map.add_mapping(
                ienv.iter().map(|i| i[1].id()),
                ienv_new.iter().map(|i| i[1].id()),
            );
        }

        // take the update terms and re-map them to use new wires
        let mut terms: Vec<Term<D, I>> = Vec::new();
        for term in &t.transition {
            terms.push(aux::map_term(term, &mut wire_map, ctx)?);
        }

        // create and push the update transition
        self.transitions
            .push(Transition::new(intf_in, intf_env, intf_out, terms));

        Ok(())
    }
}

impl<'a, D, I> IntoIterator for &'a WiredTransitions<D, I>
where
    D: 'a,
    I: 'a,
{
    type Item = &'a Transition<D, I>;
    type IntoIter = std::slice::Iter<'a, Transition<D, I>>;

    fn into_iter(self) -> Self::IntoIter {
        self.transitions.iter()
    }
}

impl<D, I> IntoIterator for WiredTransitions<D, I> {
    type Item = Transition<D, I>;
    type IntoIter = std::vec::IntoIter<Transition<D, I>>;

    fn into_iter(self) -> Self::IntoIter {
        self.transitions.into_iter()
    }
}
