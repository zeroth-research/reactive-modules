use crate::Module;
use crate::term::{Block, Term};
use crate::wire::Wire;
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use theory::{Combinatorial, Differential, Sequential, Theory};

use crate::var::{Interface, Var};
#[cfg(debug_assertions)]
use std::collections::HashSet;
use std::fmt::Debug;

//============================================================
// Atom
//============================================================

/// This data structure corresponds to the atom of reactive modules.
#[derive(Debug, Clone)]
pub struct Atom<I, J, F, S = <I as Theory>::Sort>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
    /// Corresponds to ctr wires.
    ctrl: Interface<S>,
    /// Corresponds to wait wires.
    wait: Interface<S>,
    /// Corresponds to read wires.
    read: Interface<S>,
    /// Corresponds to the initial condition.
    init: Block<I>,
    /// Corresponds to the update action.
    update: Block<J>,
    /// Corresponds to the delay activity.
    delay: Block<F>,

    /// cache of all wires local to blocks
    local: Vec<Wire<S>>,
    /// cache of all wires used in blocks
    used: Vec<Wire<S>>,
}

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
    /// Returns a reference to the initial action.
    pub fn init(&self) -> &Block<I> {
        &self.init
    }
    /// Returns a reference to the update action.
    pub fn update(&self) -> &Block<J> {
        &self.update
    }

    /// Returns a reference to the delay activity.
    pub fn delay(&self) -> &Block<F> {
        &self.delay
    }

    /// Returns a reference to the controlled variables.
    pub fn ctrl(&self) -> &Interface<S> {
        &self.ctrl
    }

    /// Returns a reference to the awaited variables.
    pub fn wait(&self) -> &Interface<S> {
        &self.wait
    }

    /// Returns a reference to the read variables.
    pub fn read(&self) -> &Interface<S> {
        &self.read
    }

    /// Returns an iterator over the temporary, local wires.
    pub(crate) fn local(&self) -> &[Wire<S>] {
        self.local.as_slice()
    }

    pub(crate) fn used(&self) -> &[Wire<S>] {
        self.used.as_slice()
    }

    /// Constructs an atom with no variables and empty blocks.
    pub fn empty() -> Self {
        Self {
            ctrl: Interface::empty(),
            wait: Interface::empty(),
            read: Interface::empty(),
            init: Block::empty(),
            update: Block::empty(),
            delay: Block::empty(),
            used: Vec::new(),
            local: Vec::new(),
        }
    }
}

impl<I, J, F, S> Default for Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
    fn default() -> Self {
        Self::empty()
    }
}

//============================================================
// Private routines
//============================================================

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
    /// Returns true if this atom awaits the other atom.
    pub fn awaits(&self, other: &Atom<I, J, F, S>) -> bool {
        !self.wait.is_disjoint(&other.ctrl)
    }

    /// Creates an atom from its components. This method checks the inputs only using assertions
    /// in debug mode.
    #[allow(clippy::too_many_arguments)]
    fn new_unchecked(
        ctrl: Interface<S>,
        wait: Interface<S>,
        read: Interface<S>,
        init: Block<I>,
        update: Block<J>,
        delay: Block<F>,
        used: Vec<Wire<S>>,
        local: Vec<Wire<S>>,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            //================================================================================
            // Check declared wires
            //================================================================================
            debug_assert!(ctrl.is_disjoint(&wait));

            //================================================================================
            // Check init block
            //================================================================================
            // all read wires are wait X wires
            let mut riter = init.read().iter();
            debug_assert!(riter.all(|w| wait.var(w).is_some_and(|v| v.nxt() == w)));
            // all write wires are ctrl X wires or local
            let mut witer = init.write().iter().filter(|w| !local.contains(w));
            debug_assert!(witer.all(|w| ctrl.var(w).is_some_and(|v| v.nxt() == w)));
            // all control X wires are written
            debug_assert!(ctrl.iter().map(Var::nxt).all(|w| init.write().contains(w)));

            //================================================================================
            // Check update block
            //================================================================================
            // all read wires are either read latched wires or wait X wires
            let mut riter = update.read().iter();
            debug_assert!(riter.all(|w| read.var(w).is_some_and(|v| v.ltc() == w)
                | wait.var(w).is_some_and(|v| v.nxt() == w)));
            // all write wires are ctrl X wires or local
            let mut witer = update.write().iter().filter(|w| !local.contains(w));
            debug_assert!(witer.all(|w| ctrl.var(w).is_some_and(|v| v.nxt() == w)));
            // all control X wires are written
            let citer = ctrl.iter();
            debug_assert!(citer.map(Var::nxt).all(|w| update.write().contains(w)));

            //================================================================================
            // Check delay block
            //================================================================================
            // all read wires are either read latched wires or wait d wires
            let mut riter = delay.read().iter();
            debug_assert!(riter.all(|w| read.var(w).is_some_and(|v| v.ltc() == w)
                | wait.var(w).is_some_and(|v| v.der() == w)));
            // all write wires are ctrl X wires or local
            let mut witer = delay.write().iter().filter(|w| !local.contains(w));
            debug_assert!(witer.all(|w| ctrl.var(w).is_some_and(|v| v.der() == w)));
            // all control d wires are written
            debug_assert!(ctrl.iter().map(Var::der).all(|w| delay.write().contains(w)));

            //================================================================================
            // Check wires and locals
            //================================================================================
            let mut found: HashSet<_> = HashSet::new();
            found.extend(init.read().iter().chain(init.write().iter()));
            found.extend(update.read().iter().chain(update.write().iter()));
            found.extend(delay.read().iter().chain(delay.write().iter()));

            // found == used
            debug_assert!(used.is_sorted());
            debug_assert!(used.windows(2).all(|w| w[0] < w[1]));
            debug_assert!(found.iter().all(|w| used.binary_search(w).is_ok()));
            debug_assert!(used.iter().all(|w| found.contains(w)));

            // found \ (ctrl U read U wait) == local
            found.retain(|w| read.var(w).is_none() & ctrl.var(w).is_none() & wait.var(w).is_none());
            debug_assert!(local.is_sorted());
            debug_assert!(local.windows(2).all(|w| w[0] < w[1]));
            debug_assert!(found.iter().all(|w| local.binary_search(w).is_ok()));
            debug_assert!(local.iter().all(|w| found.contains(w)));
        }

        Self {
            ctrl,
            wait,
            read,
            init,
            update,
            delay,
            used,
            local,
        }
    }

    fn var(&self, wire: &Wire<S>) -> Option<&Var<S>> {
        self.read
            .var(wire)
            .or_else(|| self.ctrl.var(wire))
            .or_else(|| self.wait.var(wire))
    }
}

struct WireMap<'a, S> {
    ltc: HashMap<&'a Wire<S>, &'a Var<S>>,
    nxt: HashMap<&'a Wire<S>, &'a Var<S>>,
    der: HashMap<&'a Wire<S>, &'a Var<S>>,
}

impl<'a, S> WireMap<'a, S> {
    fn unpack<V>(vars: V) -> Self
    where
        V: IntoIterator<Item = &'a Var<S>>,
    {
        let mut ltc: HashMap<&Wire<S>, &Var<S>> = HashMap::new();
        let mut nxt: HashMap<&Wire<S>, &Var<S>> = HashMap::new();
        let mut der: HashMap<&Wire<S>, &Var<S>> = HashMap::new();

        for var in vars.into_iter() {
            ltc.insert(var.ltc(), var);
            nxt.insert(var.nxt(), var);
            der.insert(var.der(), var);
        }

        Self { ltc, nxt, der }
    }
}

fn infer_ctrl<T, S: Clone>(
    block: &Block<T>,
    pool: &HashMap<&Wire<S>, &Var<S>>,
) -> Result<BTreeSet<Var<S>>, String>
where
    T: Theory<Sort = S>,
{
    // tree map on id is used to guarantee consistent order
    let mut ctrl: BTreeSet<Var<S>> = BTreeSet::new();

    for wt in block.write().iter() {
        // if the block writes to a wire in the pool of next or derived,
        // then the respective variable is controlled
        if let Some(&var) = pool.get(&wt) {
            ctrl.insert(var.clone());
        }
    }

    Ok(ctrl)
}

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Clone + Debug,
{
    fn with_ctrl(
        wmap: WireMap<S>,
        ctrl: BTreeSet<Var<S>>,
        init: Block<I>,
        update: Block<J>,
        delay: Block<F>,
    ) -> Result<Self, String> {
        let mut read: BTreeSet<Var<S>> = BTreeSet::new();
        let mut wait: BTreeSet<Var<S>> = BTreeSet::new();
        let mut used: BTreeSet<Wire<S>> = BTreeSet::new();
        let mut local: BTreeSet<Wire<S>> = BTreeSet::new();

        for rd in init.read().iter() {
            used.insert(rd.clone());
            // init can only read from next wires
            if let Some(&var) = wmap.nxt.get(&rd) {
                wait.insert(var.clone());
                continue;
            }

            if wmap.ltc.contains_key(&rd) {
                return Err(format!("Init reads latched wire {:?}", rd));
            } else if wmap.der.contains_key(&rd) {
                return Err(format!("Init reads d wire {:?}", rd));
            } else {
                return Err(format!("Init reads dangling wire {:?}", rd));
            }
        }

        for rd in update.read().iter() {
            used.insert(rd.clone());
            // if the update reads from a next wire, then this is awaited
            // otherwise, it must be read from outside the atom (dangling)
            if let Some(&var) = wmap.ltc.get(&rd) {
                read.insert(var.clone());
                continue;
            }

            if let Some(&var) = wmap.nxt.get(&rd) {
                wait.insert(var.clone());
                continue;
            }

            if wmap.der.contains_key(&rd) {
                return Err(format!("Update reads d wire {:?}", rd));
            } else {
                return Err(format!("Update reads dangling wire {:?}", rd));
            }
        }

        for rd in delay.read().iter() {
            used.insert(rd.clone());
            // if the delay reads from a derived wire, then this is awaited
            // otherwise, it must be read from outside the atom (dangling)
            if let Some(&var) = wmap.ltc.get(&rd) {
                read.insert(var.clone());
                continue;
            }

            if let Some(&var) = wmap.der.get(&rd) {
                wait.insert(var.clone());
                continue;
            }

            if wmap.nxt.contains_key(&rd) {
                return Err(format!("Delay reads X wire {:?}", rd));
            } else {
                return Err(format!("Delay reads dangling wire {:?}", rd));
            }
        }

        for wt in init.write().iter().chain(update.write().iter()) {
            used.insert(wt.clone());
            // if the init/update writes to a next wire, then this wire must be controlled
            // otherwise, it must be local
            if let Some(&var) = wmap.nxt.get(&wt) {
                if !ctrl.contains(var) {
                    return Err(format!("Init/update writes uncontrolled X wire {:?}", wt));
                }
                continue;
            }

            if wmap.ltc.contains_key(&wt) {
                return Err(format!("Init/update writes latched wire {:?}", wt));
            } else if wmap.der.contains_key(&wt) {
                return Err(format!("Init/update writes d wire {:?}", wt));
            }

            local.insert(wt.clone());
        }

        for wt in delay.write().iter() {
            used.insert(wt.clone());
            // if the delay writes to a derived wire, then this wire is controlled
            // otherwise, it must be local
            if let Some(&var) = wmap.der.get(&wt) {
                if !ctrl.contains(var) {
                    return Err(format!("Delay writes uncontrolled d wire {:?}", wt));
                }
                continue;
            }

            if wmap.ltc.contains_key(&wt) {
                return Err(format!("Delay writes latched wire {:?}", wt));
            } else if wmap.nxt.contains_key(&wt) {
                return Err(format!("Delay writes X wire {:?}", wt));
            }

            local.insert(wt.clone());
        }

        for ctr in ctrl.iter() {
            if !init.write().iter().any(|wrt| wrt == ctr.nxt()) {
                return Err(format!(
                    "Controlled X wire {:?} is not written in init",
                    ctr.nxt()
                ));
            }
            if !update.write().iter().any(|wrt| wrt == ctr.nxt()) {
                return Err(format!(
                    "Controlled X wire {:?} is not written in update",
                    ctr.nxt()
                ));
            }
            if !delay.write().iter().any(|wrt| wrt == ctr.der()) {
                return Err(format!(
                    "Controlled d wire {:?} is not written in delay",
                    ctr.der()
                ));
            }
        }

        Ok(Self::new_unchecked(
            Interface::from_exact_iter_unchecked(ctrl),
            Interface::from_exact_iter_unchecked(wait),
            Interface::from_exact_iter_unchecked(read),
            init,
            update,
            delay,
            used.into_iter().collect(),
            local.into_iter().collect(),
        ))
    }
}

//============================================================
// Public constructors
//============================================================

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Clone + Debug,
{
    /// Constructs a **sequential atom**: behaviour that evolves over discrete
    /// time steps.
    ///
    /// A sequential atom specifies an initialisation (`init`) and a discrete
    /// state update (`update`), relating the latched wires to the next wires
    /// across time steps. The delay is synthesised as `ZERO`, so the atom has
    /// no continuous dynamics.
    ///
    /// The control-related wires (`ctrl`, `wait`, `read`, and `temp`) are
    /// inferred from the variables and the terms; the controlled variables
    /// are those whose next wire is written by `init`.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom.
    /// - `init`: The terms defining the initial action of the atom.
    /// - `update`: The terms defining the discrete state update at each time step.
    ///
    /// # Returns
    /// A `Result` containing the constructed sequential atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::differential`], the continuous dual: only a delay, implicit skip update.
    /// - [`Atom::hybrid`], when both discrete and continuous dynamics are explicit.
    /// - [`Atom::combinatorial`], for time-independent, purely reactive behaviour.
    pub fn sequential<'b, V, W, U>(vars: V, init: W, update: U) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'b Var<S>>,
        W: IntoIterator<Item = Term<I>>,
        U: IntoIterator<Item = Term<J>>,
        S: 'b,
    {
        let wires = WireMap::unpack(vars);

        let init = Block::try_from_iter(init.into_iter().map(Ok))?;
        let ctrl = infer_ctrl(&init, &wires.nxt)?;
        let ctrl_der = ctrl.iter().map(Var::der).cloned();

        let update = Block::try_from_iter(update.into_iter().map(Ok))?;
        let delay = Block::zero(ctrl_der)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **differential atom**: behaviour that evolves continuously.
    ///
    /// A differential atom specifies an initialisation (`init`) and a
    /// continuous evolution (`delay`) driving the derivative wires. The
    /// discrete update is synthesised as one `SKIP` term per controlled
    /// variable, copying the latched value to the next wire, so all change
    /// comes from the continuous dynamics.
    ///
    /// The control-related wires (`ctrl`, `wait`, `read`, and `temp`) are
    /// inferred from the variables and the terms; the controlled variables
    /// are those whose next wire is written by `init`.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom.
    /// - `init`: The terms defining the initial action of the atom.
    /// - `delay`: The terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed differential atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::sequential`], the discrete dual: only an update, zero delay.
    /// - [`Atom::hybrid`], when both discrete and continuous dynamics are explicit.
    pub fn differential<'b, V, W, Z>(vars: V, init: W, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'b Var<S>>,
        W: IntoIterator<Item = Term<I>>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'b,
    {
        let wires = WireMap::unpack(vars);

        let init = Block::try_from_iter(init.into_iter().map(Ok))?;
        let ctrl = infer_ctrl(&init, &wires.nxt)?;
        let ctrl_nxt = ctrl.iter().map(Var::nxt).cloned();
        let ctrl_ltc = ctrl.iter().map(Var::ltc).cloned();

        let update = Block::skip(ctrl_nxt, ctrl_ltc)?;
        let delay = Block::try_from_iter(delay.into_iter().map(Ok))?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **hybrid atom**, combining discrete and continuous dynamics.
    ///
    /// A hybrid atom specifies all three blocks explicitly: an initialisation
    /// (`init`), a discrete state update (`update`), and a continuous
    /// evolution (`delay`) driving the derivative wires. It is the most
    /// general constructor; [`Atom::sequential`] and [`Atom::differential`]
    /// are the special cases where the delay (resp. the update) is
    /// synthesised automatically.
    ///
    /// The control-related wires (`ctrl`, `wait`, `read`, and `temp`) are
    /// inferred from the variables and the terms; the controlled variables
    /// are those whose next wire is written by `init`.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom.
    /// - `init`: The terms defining the initial action of the atom.
    /// - `update`: The terms defining the discrete state update at each time step.
    /// - `delay`: The terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed hybrid atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::sequential`], for purely discrete behaviour (delay is zero).
    /// - [`Atom::differential`], for purely continuous behaviour (update is skip).
    pub fn hybrid<'a, V, W, U, Z>(vars: V, init: W, update: U, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Var<S>>,
        W: IntoIterator<Item = Term<I>>,
        U: IntoIterator<Item = Term<J>>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars);

        let init = Block::try_from_iter(init.into_iter().map(Ok))?;
        let update = Block::try_from_iter(update.into_iter().map(Ok))?;
        let delay = Block::try_from_iter(delay.into_iter().map(Ok))?;

        let ctrl = infer_ctrl(&init, &wires.nxt)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **jump atom**: sequential behaviour whose
    /// initial state is left unconstrained.
    ///
    /// Only the update is given; the controlled variables are inferred from
    /// the next wires it writes. The initialisation is synthesised as one
    /// `HAVOC` term per controlled variable, so their initial values are
    /// arbitrary, and the delay is synthesised as `ZERO`, so the atom has no
    /// continuous dynamics.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom.
    /// - `update`: The terms defining the discrete state update at each time step.
    ///
    /// # Returns
    /// A `Result` containing the constructed atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::sequential`], when the initial state is constrained explicitly.
    /// - [`Atom::constant`], the dual: only the initial state, no update.
    pub fn jump<'a, V, U>(vars: V, update: U) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Var<S>>,
        U: IntoIterator<Item = Term<J>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars);

        let update = Block::try_from_iter(update.into_iter().map(Ok))?;
        let ctrl = infer_ctrl(&update, &wires.nxt)?;
        let ctrl_nxt = ctrl.iter().map(Var::nxt).cloned();
        let ctrl_der = ctrl.iter().map(Var::der).cloned();

        let init = Block::havoc(ctrl_nxt)?;
        let delay = Block::zero(ctrl_der)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs an **uninitialized atom**: discrete and continuous
    /// dynamics with an unconstrained initial state.
    ///
    /// The discrete update and the continuous delay are given; the
    /// controlled variables are inferred from the next wires the update
    /// writes, and the initialisation is synthesised as one `HAVOC` term per
    /// controlled variable, so the initial values are arbitrary.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom.
    /// - `update`: The terms defining the discrete state update at each time step.
    /// - `delay`: The terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::hybrid`], when the initial state is constrained explicitly.
    /// - [`Atom::jump`] and [`Atom::flow`], the purely discrete and purely
    ///   continuous special cases.
    pub fn uninitialized<'a, V, U, Z>(vars: V, update: U, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Var<S>>,
        U: IntoIterator<Item = Term<J>>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars);

        let update = Block::try_from_iter(update.into_iter().map(Ok))?;
        let ctrl = infer_ctrl(&update, &wires.nxt)?;
        let ctrl_nxt = ctrl.iter().map(Var::nxt).cloned();

        let init = Block::havoc(ctrl_nxt)?;
        let delay = Block::try_from_iter(delay.into_iter().map(Ok))?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **constant atom**: variables that are set once at
    /// initialisation and then hold their value forever.
    ///
    /// Only the initialisation is given; the controlled variables are
    /// inferred from the next wires it writes. The update is synthesised as
    /// one `SKIP` term per controlled variable, copying the latched value to
    /// the next wire at every step, and the delay is synthesised as `ZERO`,
    /// so the values never drift.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom.
    /// - `init`: The terms defining the (only) assignment of the variables.
    ///
    /// # Returns
    /// A `Result` containing the constructed atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::jump`], the dual: only an update, arbitrary initial state.
    /// - [`Atom::sequential`], when the variables also evolve over time.
    pub fn constant<'a, V, W>(vars: V, init: W) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Var<S>>,
        W: IntoIterator<Item = Term<I>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars);

        let init = Block::try_from_iter(init.into_iter().map(Ok))?;
        let ctrl = infer_ctrl(&init, &wires.nxt)?;
        let ctrl_ltc = ctrl.iter().map(Var::ltc).cloned();
        let ctrl_nxt = ctrl.iter().map(Var::nxt).cloned();
        let ctrl_der = ctrl.iter().map(Var::der).cloned();

        let update = Block::skip(ctrl_nxt, ctrl_ltc)?;
        let delay = Block::zero(ctrl_der)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **hold atom**: variables that hold an arbitrary but
    /// fixed value — an adversarial constant.
    ///
    /// The atom takes no blocks: it controls every variable in `vars`, with
    /// all three blocks synthesised per variable — `HAVOC` init, `SKIP`
    /// update, `ZERO` delay. Each value is chosen nondeterministically at
    /// initialisation and never changes: this models free parameters, i.e.,
    /// quantities a system depends on but does not compute.
    ///
    /// # Parameters
    /// - `vars`: The variables of the atom, all of which it controls.
    ///
    /// # Returns
    /// A `Result` containing the constructed hold atom if successful,
    /// or an error string if consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::constant`], when the held value is set explicitly at initialisation.
    /// - [`Atom::jump`], when the variables also evolve over time.
    /// - [`Module::hold`], for the module-level shorthand.
    pub fn hold<'a, V>(vars: V) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Var<S>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars);

        // every variable is controlled
        let ctrl: BTreeSet<Var<S>> = wires.ltc.values().cloned().cloned().collect();
        let ctrl_ltc = ctrl.iter().map(Var::ltc).cloned();
        let ctrl_nxt = ctrl.iter().map(Var::nxt).cloned();
        let ctrl_der = ctrl.iter().map(Var::der).cloned();

        let init = Block::havoc(ctrl_nxt.clone())?;
        let update = Block::skip(ctrl_nxt, ctrl_ltc)?;
        let delay = Block::zero(ctrl_der)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **flow atom**: purely continuous behaviour with an
    /// unconstrained initial state.
    ///
    /// Only the continuous evolution (`delay`) is given; the controlled
    /// variables are inferred from the derivative wires it writes. The
    /// initialisation is synthesised as one `HAVOC` term per controlled
    /// variable, so the initial values are arbitrary, and the update as
    /// `SKIP`, so all change comes from the continuous dynamics.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom.
    /// - `delay`: The terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed flow atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::differential`], when the initial state is constrained explicitly.
    /// - [`Atom::hybrid`], when discrete dynamics are involved as well.
    pub fn flow<'a, V, Z>(vars: V, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Var<S>>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars);

        let delay = Block::try_from_iter(delay.into_iter().map(Ok))?;
        let ctrl = infer_ctrl(&delay, &wires.der)?;
        let ctrl_ltc = ctrl.iter().map(Var::ltc).cloned();
        let ctrl_nxt = ctrl.iter().map(Var::nxt).cloned();

        let init = Block::havoc(ctrl_nxt.clone())?;
        let update = Block::skip(ctrl_nxt, ctrl_ltc)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **combinatorial atom**: purely reactive behaviour without
    /// temporal state.
    ///
    /// A combinatorial atom specifies a single block of assignments (`assign`)
    /// relating next wires *within the same time step*. The same block serves
    /// as both initialisation and update, and the delay is synthesised as
    /// `ZERO`, so the atom is time-independent.
    ///
    /// The control-related wires (`ctrl`, `wait`, `read`, and `temp`) are
    /// inferred from the variables and the terms; the controlled variables
    /// are those whose next wire is written by `assign`.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom.
    /// - `assign`: The terms defining the combinatorial relationships between next wires.
    ///
    /// # Returns
    /// A `Result` containing the constructed combinatorial atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::sequential`], the stateful dual with explicit time evolution.
    /// - [`Module::combinatorial`], for combinatorial modules.
    pub fn combinatorial<'a, T, V, W>(vars: V, assign: W) -> Result<Self, String>
    where
        T: Theory<Sort = S> + Into<I> + Into<J> + Clone,
        V: IntoIterator<Item = &'a Var<S>>,
        W: IntoIterator<Item = Term<T>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars);

        let assign: Block<T> = Block::try_from_iter(assign.into_iter().map(Ok))?;
        let ctrl = infer_ctrl(&assign, &wires.nxt)?;

        let init: Block<I> = Block::convert(assign.clone());
        let update: Block<J> = Block::convert(assign);
        let delay: Block<F> = Block::zero(ctrl.iter().map(Var::der).cloned())?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }
}

//============================================================
// Display routines
//============================================================

pub struct Display<'a, I, J, F, S, N>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    N: Fn(&Var<S>) -> Cow<'a, str>,
{
    atom: &'a Atom<I, J, F, S>,
    name: N,
    typed: bool,
}

impl<'a, I, J, F, S, N> Display<'a, I, J, F, S, N>
where
    I: Combinatorial<Sort = S> + fmt::Display,
    J: Sequential<Sort = S> + fmt::Display,
    F: Differential<Sort = S> + fmt::Display,
    S: fmt::Display,
    N: Fn(&Var<S>) -> Cow<'a, str>,
{
    fn fmt_vars<V: Iterator<Item = &'a Var<S>>>(
        &self,
        f: &mut fmt::Formatter<'_>,
        iter: V,
    ) -> fmt::Result {
        for (i, var) in iter.enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if self.typed {
                write!(f, "{}", var.with_name((self.name)(var)))?;
            } else {
                write!(f, "{}", (self.name)(var))?;
            }
        }
        Ok(())
    }

    fn fmt_wirename(&self, wire: &Wire<S>) -> Cow<'a, str> {
        if let Some(var) = self.atom.var(wire) {
            if wire == var.ltc() {
                // the plain name passes through as-is: borrowed stays borrowed
                (self.name)(var)
            } else if wire == var.nxt() {
                Cow::Owned(format!("X({})", (self.name)(var)))
            } else {
                assert!(wire == var.der());
                Cow::Owned(format!("d({})", (self.name)(var)))
            }
        } else {
            Cow::Owned(format!("{}", wire.untyped()))
        }
    }

    fn fmt_terms<T: 'a + Theory<Sort = S> + fmt::Display, U: Iterator<Item = &'a Term<T>>>(
        &self,
        f: &mut fmt::Formatter<'_>,
        iter: U,
        pad: &str,
    ) -> fmt::Result {
        // the argument annotation makes the closure higher-ranked over the
        // wire's lifetime, as `with_wirenames`'s `Fn` bound requires
        let name = |w: &Wire<S>| self.fmt_wirename(w);

        for term in iter {
            let display = term.with_wirenames(name);
            writeln!(f, "{pad}{}", display)?;
        }
        Ok(())
    }

    pub fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, pad: &str) -> fmt::Result {
        const BOLD: &str = "\x1b[1m";
        const RESET: &str = "\x1b[0m";
        const INDENT: &str = "  ";

        let a = self.atom;
        write!(f, "{pad}{BOLD}atom{RESET}")?;
        write!(f, " {BOLD}controls{RESET} ")?;
        self.fmt_vars(f, a.ctrl.iter())?;
        if !a.read.is_empty() {
            write!(f, " {BOLD}reads{RESET} ")?;
            self.fmt_vars(f, a.read.iter())?;
        }
        if !a.wait.is_empty() {
            write!(f, " {BOLD}awaits{RESET} ")?;
            self.fmt_vars(f, a.wait.iter())?;
        }
        let tpad = format!("{pad}{INDENT}");
        writeln!(f, "\n{pad}{BOLD}init{RESET}")?;
        self.fmt_terms(f, a.init.iter(), tpad.as_str())?;
        writeln!(f, "{pad}{BOLD}delay{RESET}")?;
        self.fmt_terms(f, a.delay.iter(), tpad.as_str())?;
        writeln!(f, "{pad}{BOLD}update{RESET}")?;
        self.fmt_terms(f, a.update.iter(), tpad.as_str())?;

        Ok(())
    }
}

impl<'a, I, J, F, S, N> fmt::Display for Display<'a, I, J, F, S, N>
where
    I: Combinatorial<Sort = S> + fmt::Display,
    J: Sequential<Sort = S> + fmt::Display,
    F: Differential<Sort = S> + fmt::Display,
    S: fmt::Display,
    N: Fn(&Var<S>) -> Cow<'a, str>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.fmt_indent(f, "")
    }
}

impl<I, J, D, S> Atom<I, J, D, S>
where
    I: Combinatorial<Sort = S> + fmt::Display,
    J: Sequential<Sort = S> + fmt::Display,
    D: Differential<Sort = S> + fmt::Display,
    S: fmt::Display,
{
    pub fn with_varnames<'a, N, R>(
        &'a self,
        name: N,
    ) -> Display<'a, I, J, D, S, impl Fn(&Var<S>) -> Cow<'a, str>>
    where
        N: Fn(&Var<S>) -> R,
        R: Into<Cow<'a, str>>,
    {
        Display {
            atom: self,
            name: move |v: &Var<S>| name(v).into(),
            typed: true,
        }
    }

    pub fn with_varnames_untyped<'a, N, R>(
        &'a self,
        name: N,
    ) -> Display<'a, I, J, D, S, impl Fn(&Var<S>) -> Cow<'a, str>>
    where
        N: Fn(&Var<S>) -> R,
        R: Into<Cow<'a, str>>,
    {
        Display {
            atom: self,
            name: move |v: &Var<S>| name(v).into(),
            typed: false,
        }
    }
}

//============================================================
// Atomic modules
//============================================================

impl<I, J, F, S> Module<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Clone + Debug,
{
    /// Constructs a **sequential module**: behaviour that evolves over
    /// discrete time steps.
    ///
    /// A sequential module specifies an initialisation and a discrete state
    /// update; it has no continuous dynamics. It is composed of a single
    /// [`Atom::sequential`] atom over `vars`, and is **fully
    /// observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module.
    /// - `init`: The set of terms defining the module's initial state.
    /// - `update`: The set of terms defining the module's state update at each time step.
    ///
    /// # Returns
    /// A `Result` containing the constructed sequential module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::differential`] and [`Module::hybrid`], when continuous
    ///   dynamics are involved.
    /// - [`Module::combinatorial`], for stateless, time-independent modules.
    /// - [`Atom::sequential`], for creating individual sequential atoms.
    pub fn sequential<'a, V, W, U>(vars: V, init: W, update: U) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Var<S>>,
        W: IntoIterator<Item = Term<I>>,
        U: IntoIterator<Item = Term<J>>,
        S: 'a,
    {
        let atom = Atom::sequential(vars, init, update)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs a **differential module**: behaviour that evolves
    /// continuously.
    ///
    /// A differential module specifies an initialisation and a continuous
    /// evolution of the derivatives; the discrete update is an implicit skip.
    /// It is composed of a single [`Atom::differential`] atom over `vars`, and is **fully
    /// observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module.
    /// - `init`: The set of terms defining the module's initial state.
    /// - `delay`: The set of terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed differential module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::sequential`], the discrete dual: only an update, no continuous dynamics.
    /// - [`Module::hybrid`], when both discrete and continuous dynamics are explicit.
    /// - [`Atom::differential`], for creating individual differential atoms.
    pub fn differential<'a, V, W, Z>(vars: V, init: W, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Var<S>>,
        W: IntoIterator<Item = Term<I>>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let atom = Atom::differential(vars, init, delay)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs a **hybrid module**, combining discrete updates with
    /// continuous dynamics.
    ///
    /// A hybrid module specifies all three blocks explicitly — initialisation,
    /// discrete update, and continuous delay. It is composed of a single
    /// [`Atom::hybrid`] atom over `vars`, and is **fully
    /// observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module.
    /// - `init`: The set of terms defining the module's initial state.
    /// - `update`: The set of terms defining the discrete state update.
    /// - `delay`: The set of terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed hybrid module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::sequential`] and [`Module::differential`], the purely
    ///   discrete and purely continuous special cases.
    /// - [`Atom::hybrid`], for creating individual hybrid atoms.
    pub fn hybrid<'a, V, W, U, Z>(vars: V, init: W, update: U, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Var<S>>,
        W: IntoIterator<Item = Term<I>>,
        U: IntoIterator<Item = Term<J>>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let atom = Atom::hybrid(vars, init, update, delay)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs a **jump module**: sequential behaviour whose
    /// initial state is left unconstrained.
    ///
    /// Only the update is given; the initial values of the controlled
    /// variables are havoced. It is composed of a single
    /// [`Atom::jump`] atom over `vars`, and is **fully
    /// observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module.
    /// - `update`: The set of terms defining the module's state update at each time step.
    ///
    /// # Returns
    /// A `Result` containing the constructed module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::sequential`], when the initial state is constrained explicitly.
    /// - [`Module::constant`], the dual: only the initial state, no update.
    pub fn jump<'a, V, U>(vars: V, update: U) -> Result<Self, String>
    where
        U: IntoIterator<Item = Term<J>>,
        V: IntoIterator<Item = &'a Var<S>>,
        S: 'a,
    {
        let atom = Atom::jump(vars, update)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs an **uninitialized module**: discrete and continuous
    /// dynamics with an unconstrained initial state.
    ///
    /// The discrete update and the continuous delay are given; the initial
    /// values of the controlled variables are havoced. It is composed of a
    /// single [`Atom::uninitialized`] atom over `vars`, and is **fully
    /// observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module.
    /// - `update`: The set of terms defining the discrete state update.
    /// - `delay`: The set of terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::hybrid`], when the initial state is constrained explicitly.
    /// - [`Module::jump`] and [`Module::flow`], the purely discrete and
    ///   purely continuous special cases.
    pub fn uninitialized<'a, V, U, Z>(vars: V, update: U, delay: Z) -> Result<Self, String>
    where
        U: IntoIterator<Item = Term<J>>,
        Z: IntoIterator<Item = Term<F>>,
        V: IntoIterator<Item = &'a Var<S>>,
        S: 'a,
    {
        let atom = Atom::uninitialized(vars, update, delay)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs a **hold module**: variables that hold an arbitrary but
    /// fixed value — symbolic constants.
    ///
    /// The module takes no blocks: each value is chosen nondeterministically
    /// at initialisation and never changes, modelling free parameters of a
    /// composition. It is composed of a single [`Atom::hold`] atom, and is
    /// **fully observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module.
    ///
    /// # Returns
    /// A `Result` containing the constructed hold module if successful,
    /// or an error string if consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::constant`], when the held value is set explicitly at initialisation.
    /// - [`Module::jump`], when the variables also evolve over time.
    /// - [`Atom::hold`], for creating individual hold atoms.
    pub fn hold<'a, V>(vars: V) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Var<S>>,
        S: 'a,
    {
        let atom = Atom::hold(vars)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs a **flow module**: purely continuous behaviour with an
    /// unconstrained initial state.
    ///
    /// Only the continuous evolution of the derivatives is given; the initial
    /// values are havoced and the discrete update is an implicit skip. It is
    /// composed of a single [`Atom::flow`] atom over `vars`, and is **fully
    /// observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module.
    /// - `delay`: The set of terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed flow module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::differential`], when the initial state is constrained explicitly.
    /// - [`Atom::flow`], for creating individual flow atoms.
    pub fn flow<'a, V, Z>(vars: V, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Var<S>>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let atom = Atom::flow(vars, delay)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs a **constant module**: variables that are set once at
    /// initialisation and then hold their value forever.
    ///
    /// Only the initialisation is given; the variables are held by an implicit
    /// skip update and have no continuous dynamics. It is composed of a single
    /// [`Atom::constant`] atom, and is **fully observable**: use
    /// [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module.
    /// - `init`: The set of terms defining the (only) assignment of the variables.
    ///
    /// # Returns
    /// A `Result` containing the constructed module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::jump`], the dual: only an update, arbitrary initial state.
    /// - [`Atom::constant`], for creating individual constant atoms.
    pub fn constant<'a, V, W>(vars: V, init: W) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Var<S>>,
        W: IntoIterator<Item = Term<I>>,
        S: 'a,
    {
        let atom = Atom::constant(vars, init)?;
        Self::observable(std::iter::once(atom))
    }

    /// Constructs a **combinatorial module**: a stateless, time-independent
    /// relationship between observable wires.
    ///
    /// A combinatorial module specifies a single block of assignments
    /// computing the outputs from the inputs within the same time step. It is
    /// composed of a single [`Atom::combinatorial`] atom, and is **fully
    /// observable**: use [`Module::hide`] to make variables private.
    ///
    /// # Parameters
    /// - `vars`: The variables of the module.
    /// - `assign`: The set of combinatorial assignment terms defining how the
    ///   output is computed from the input.
    ///
    /// # Returns
    /// A `Result` containing the constructed combinatorial module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::sequential`], the stateful dual with explicit time evolution.
    /// - [`Atom::combinatorial`], for creating individual combinatorial atoms.
    pub fn combinatorial<'a, T, V, W>(vars: V, assign: W) -> Result<Self, String>
    where
        T: Theory<Sort = S> + Into<I> + Into<J> + Clone,
        V: IntoIterator<Item = &'a Var<S>>,
        W: IntoIterator<Item = Term<T>>,
        S: 'a,
    {
        let atom = Atom::combinatorial(vars, assign)?;
        Self::observable(std::iter::once(atom))
    }
}
