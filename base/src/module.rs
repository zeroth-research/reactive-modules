use crate::atom::Atom;
use crate::wire::Wire;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

/// This data structure corresponds to the module of reactive modules.
#[derive(Debug)]
pub struct Module<D, I> {
    /// Correspond to the wires of the module divided by visibility
    /// ```text
    ///     *====================*
    ///     | extl | intf | prvt |
    ///     *--------------------*
    ///     |      |    ctrl     |
    ///     *--------------------*
    ///     |     obs     |      |
    ///     *--------------------*
    ///     |        wire        |
    ///     *====================*
    /// ```
    ///  Wires are organised in pairs of identical twins where
    ///  - 0: latched wires
    ///  - 1: next wires
    extl: [Wire<D>; 2],
    intf: [Wire<D>; 2],
    prvt: [Wire<D>; 2],
    ctrl: [Wire<D>; 2],
    obs: [Wire<D>; 2],
    wire: [Wire<D>; 2],

    /// The atoms of this module.
    /// The atoms must be stored in a *consistent* linear order
    /// as defined in the reactive modules paper.
    atoms: Vec<Atom<D, I>>,
}

impl<D: Clone + Eq + Debug, I> Module<D, I> {
    pub fn new_unchecked(
        extl: [Wire<D>; 2],
        intf: [Wire<D>; 2],
        prvt: [Wire<D>; 2],
        ctrl: [Wire<D>; 2],
        obs: [Wire<D>; 2],
        wire: [Wire<D>; 2],
        atoms: Vec<Atom<D, I>>,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(extl[0].size(), extl[1].size());
            debug_assert_eq!(intf[0].size(), intf[1].size());
            debug_assert_eq!(prvt[0].size(), prvt[1].size());
            debug_assert_eq!(ctrl[0].size(), ctrl[1].size());
            debug_assert_eq!(obs[0].size(), obs[1].size());
            debug_assert_eq!(wire[0].size(), wire[1].size());

            let mut ltch_to_dtype: HashMap<usize, &D> = HashMap::new();
            let mut next_to_ltch: HashMap<usize, usize> = HashMap::new();
            for ((a, at), (b, bt)) in wire[0].iter().zip(wire[1].iter()) {
                debug_assert_eq!(at, bt);
                // check that indices are unique, and store them
                debug_assert!(ltch_to_dtype.insert(a, &at).is_none());
                debug_assert!(next_to_ltch.insert(b, a).is_none());
            }

            let mut obs_ltch: HashSet<usize> = HashSet::new();
            for ((a, at), (b, bt)) in obs[0].iter().zip(obs[1].iter()) {
                debug_assert_eq!(at, bt);
                // check consistency with other wires
                debug_assert_eq!(ltch_to_dtype.get(&a), Some(&at));
                debug_assert_eq!(next_to_ltch.get(&b), Some(&a));
                // check that indices are unique, and store them
                debug_assert!(obs_ltch.insert(a));
            }

            let mut ctrl_ltch: HashSet<usize> = HashSet::new();
            for ((a, at), (b, bt)) in ctrl[0].iter().zip(ctrl[1].iter()) {
                debug_assert_eq!(at, bt);
                // check consistency with other wires
                debug_assert_eq!(ltch_to_dtype.get(&a), Some(&at));
                debug_assert_eq!(next_to_ltch.get(&b), Some(&a));
                // check that indices are unique, and store them
                debug_assert!(ctrl_ltch.insert(a));
            }

            let mut prvt_ltch: HashSet<usize> = HashSet::new();
            for ((a, at), (b, bt)) in prvt[0].iter().zip(prvt[1].iter()) {
                debug_assert_eq!(at, bt);
                // check consistency with other wires
                debug_assert_eq!(ltch_to_dtype.get(&a), Some(&at));
                debug_assert_eq!(next_to_ltch.get(&b), Some(&a));
                debug_assert!(!obs_ltch.contains(&a));
                debug_assert!(ctrl_ltch.contains(&a));
                // check that indices are unique, and store them
                debug_assert!(prvt_ltch.insert(a));
            }

            let mut intf_ltch: HashSet<usize> = HashSet::new();
            for ((a, at), (b, bt)) in intf[0].iter().zip(intf[1].iter()) {
                debug_assert_eq!(at, bt);
                // check consistency with other wires
                debug_assert_eq!(ltch_to_dtype.get(&a), Some(&at));
                debug_assert_eq!(next_to_ltch.get(&b), Some(&a));
                debug_assert!(obs_ltch.contains(&a));
                debug_assert!(ctrl_ltch.contains(&a));
                debug_assert!(!prvt_ltch.contains(&a));
                // check that indices are unique, and store them
                debug_assert!(intf_ltch.insert(a));
            }

            let mut extl_ltch: HashSet<usize> = HashSet::new();
            for ((a, at), (b, bt)) in extl[0].iter().zip(extl[1].iter()) {
                debug_assert_eq!(at, bt);
                // check consistency with other wires
                debug_assert_eq!(ltch_to_dtype.get(&a), Some(&at));
                debug_assert_eq!(next_to_ltch.get(&b), Some(&a));
                debug_assert!(obs_ltch.contains(&a));
                debug_assert!(!ctrl_ltch.contains(&a));
                debug_assert!(!prvt_ltch.contains(&a));
                debug_assert!(!intf_ltch.contains(&a));
                // check that indices are unique, and store them
                debug_assert!(extl_ltch.insert(a));
            }

            // check that extl, intf, and prvt contain obs, ctrl, and wire
            for (a, _) in ctrl[0].iter() {
                debug_assert!(intf_ltch.contains(&a) || prvt_ltch.contains(&a));
            }
            for (a, _) in obs[0].iter() {
                debug_assert!(extl_ltch.contains(&a) || intf_ltch.contains(&a));
            }
            for (a, _) in wire[0].iter() {
                debug_assert!(
                    extl_ltch.contains(&a) || intf_ltch.contains(&a) || prvt_ltch.contains(&a)
                );
            }

            let next_to_dtype_get = |w| next_to_ltch.get(&w).and_then(|z| ltch_to_dtype.get(z));

            // check atoms consistency
            let mut written: HashSet<usize> = HashSet::from_iter(extl[1].iter().map(|(w, _)| w));
            for (i, atom1) in atoms.iter().enumerate() {
                for (a, at) in atom1.read.iter() {
                    // reads are latched, and dtype matches
                    debug_assert_eq!(Some(&at), ltch_to_dtype.get(&a));
                }
                for (a, at) in atom1.wait.iter() {
                    // awaits are next, and dtype matches
                    debug_assert_eq!(Some(&at), next_to_dtype_get(a));
                    // await order is consistent
                    debug_assert!(written.contains(&a));
                }
                for (a, at) in atom1.ctrl.iter() {
                    // controls are next, and dtype matches
                    debug_assert_eq!(Some(&at), next_to_dtype_get(a));
                    // controls are disjoint
                    debug_assert!(written.insert(a));
                }
            }

            // check that all module control wires are written/controlled by an atom
            for (a, _) in ctrl[1].iter() {
                debug_assert!(written.contains(&a));
            }

            // TODO check that temporaries are decoupled from module wires
        }

        Module {
            extl,
            intf,
            prvt,
            ctrl,
            obs,
            wire,
            atoms,
        }
    }

    pub fn with_atoms(wire: [Wire<D>; 2], atoms: Vec<Atom<D, I>>) -> Result<Self, &'static str> {
        // Check and store wire index + dtype information
        if wire[0].size() != wire[1].size() {
            return Err("len mismatch in latched and next wires");
        }
        let mut ltch_to_dtype: HashMap<usize, &D> = HashMap::new();
        let mut next_to_ltch: HashMap<usize, usize> = HashMap::new();
        for ((a, at), (b, bt)) in wire[0].iter().zip(wire[1].iter()) {
            if at != bt {
                return Err("dtype mismatch in latched and next wires");
            }
            if ltch_to_dtype.insert(a, &at).is_some() {
                return Err("duplicate latched wire");
            }
            if next_to_ltch.insert(b, a).is_some() {
                return Err("duplicate next wire");
            }
        }

        // Check atoms consistency and infer control wires
        let mut ctrl_set: HashSet<usize> = HashSet::new();
        for (i, atom) in atoms.iter().enumerate() {
            for (a, at) in atom.read.iter() {
                if ltch_to_dtype.get(&a) != Some(&at) {
                    return Err("atom read not latched or dtype mismatch");
                }
            }
            for (a, at) in atom.wait.iter() {
                if next_to_ltch.get(&a).and_then(|i| ltch_to_dtype.get(i)) != Some(&at) {
                    return Err("atom await not next or dtype mismatch");
                }
            }
            for (a, at) in atom.ctrl.iter() {
                if next_to_ltch.get(&a).and_then(|i| ltch_to_dtype.get(i)) != Some(&at) {
                    return Err("atom control not next or dtype mismatch");
                }
                if !ctrl_set.insert(a) {
                    return Err("shared or duplicate atom control wire");
                }
            }

            for past_atom in atoms.iter().take(i) {
                if past_atom.awaits(atom) {
                    return Err("inconsistent awaiting order");
                }
            }
        }

        // Build ctrl and extl wires based on inferred control set
        let mut ctrl_0: Vec<(usize, D)> = Vec::with_capacity(ctrl_set.len());
        let mut ctrl_1: Vec<(usize, D)> = Vec::with_capacity(ctrl_set.len());
        let mut extl_0: Vec<(usize, D)> = Vec::with_capacity(wire[0].size() - ctrl_set.len());
        let mut extl_1: Vec<(usize, D)> = Vec::with_capacity(wire[0].size() - ctrl_set.len());

        for (a, at) in wire[1].iter() {
            if ctrl_set.contains(&a) {
                ctrl_0.push((next_to_ltch.get(&a).unwrap().clone(), at.clone()));
                ctrl_1.push((a, at.clone()));
            } else {
                extl_0.push((next_to_ltch.get(&a).unwrap().clone(), at.clone()));
                extl_1.push((a, at.clone()));
            }
        }

        // Build wire pairs
        let extl = [Wire::new_unchecked(extl_0), Wire::new_unchecked(extl_1)];
        let ctrl = [Wire::new_unchecked(ctrl_0), Wire::new_unchecked(ctrl_1)];
        let prvt = [Wire::none(), Wire::none()];
        let intf = ctrl.clone();
        let obs = wire.clone();

        Ok(Self::new_unchecked(
            extl, intf, prvt, ctrl, obs, wire, atoms,
        ))
    }
}
