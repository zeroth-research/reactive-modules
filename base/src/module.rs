use crate::atom::Atom;
use crate::term::Term;
use crate::wire::Wire;
use std::collections::{HashMap, HashSet};
use std::fmt;
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
            debug_assert_eq!(extl[0].len(), extl[1].len());
            debug_assert_eq!(intf[0].len(), intf[1].len());
            debug_assert_eq!(prvt[0].len(), prvt[1].len());
            debug_assert_eq!(ctrl[0].len(), ctrl[1].len());
            debug_assert_eq!(obs[0].len(), obs[1].len());
            debug_assert_eq!(wire[0].len(), wire[1].len());

            let mut ltc_to_dtype: HashMap<usize, &D> = HashMap::new();
            let mut nxt_to_ltc: HashMap<usize, usize> = HashMap::new();
            for ((ltc, dtype), (nxt, ntype)) in wire[0].iter().zip(wire[1].iter()) {
                debug_assert_eq!(dtype, ntype);
                // check that indices are unique, and store them
                debug_assert!(ltc_to_dtype.insert(ltc, dtype).is_none());
                debug_assert!(nxt_to_ltc.insert(nxt, ltc).is_none());
            }

            let mut obs_ltc: HashSet<usize> = HashSet::new();
            for ((ltc, dtype), (nxt, ntype)) in obs[0].iter().zip(obs[1].iter()) {
                debug_assert_eq!(dtype, ntype);
                // check consistency with other wires
                debug_assert_eq!(ltc_to_dtype.get(&ltc), Some(&dtype));
                debug_assert_eq!(nxt_to_ltc.get(&nxt), Some(&ltc));
                // check that indices are unique, and store them
                debug_assert!(obs_ltc.insert(ltc));
            }

            let mut ctrl_ltc: HashSet<usize> = HashSet::new();
            for ((ltc, dtype), (nxt, ntype)) in ctrl[0].iter().zip(ctrl[1].iter()) {
                debug_assert_eq!(dtype, ntype);
                // check consistency with other wires
                debug_assert_eq!(ltc_to_dtype.get(&ltc), Some(&dtype));
                debug_assert_eq!(nxt_to_ltc.get(&nxt), Some(&ltc));
                // check that indices are unique, and store them
                debug_assert!(ctrl_ltc.insert(ltc));
            }

            let mut prvt_ltc: HashSet<usize> = HashSet::new();
            for ((ltc, dtype), (nxt, ntype)) in prvt[0].iter().zip(prvt[1].iter()) {
                debug_assert_eq!(dtype, ntype);
                // check consistency with other wires
                debug_assert_eq!(ltc_to_dtype.get(&ltc), Some(&dtype));
                debug_assert_eq!(nxt_to_ltc.get(&nxt), Some(&ltc));
                debug_assert!(!obs_ltc.contains(&ltc));
                debug_assert!(ctrl_ltc.contains(&ltc));
                // check that indices are unique, and store them
                debug_assert!(prvt_ltc.insert(ltc));
            }

            let mut intf_ltc: HashSet<usize> = HashSet::new();
            for ((ltc, dtype), (nxt, ntype)) in intf[0].iter().zip(intf[1].iter()) {
                debug_assert_eq!(dtype, ntype);
                // check consistency with other wires
                debug_assert_eq!(ltc_to_dtype.get(&ltc), Some(&dtype));
                debug_assert_eq!(nxt_to_ltc.get(&nxt), Some(&ltc));
                debug_assert!(obs_ltc.contains(&ltc));
                debug_assert!(ctrl_ltc.contains(&ltc));
                debug_assert!(!prvt_ltc.contains(&ltc));
                // check that indices are unique, and store them
                debug_assert!(intf_ltc.insert(ltc));
            }

            let mut extl_ltc: HashSet<usize> = HashSet::new();
            for ((ltc, dtype), (nxt, ntype)) in extl[0].iter().zip(extl[1].iter()) {
                debug_assert_eq!(dtype, ntype);
                // check consistency with other wires
                debug_assert_eq!(ltc_to_dtype.get(&ltc), Some(&dtype));
                debug_assert_eq!(nxt_to_ltc.get(&nxt), Some(&ltc));
                debug_assert!(obs_ltc.contains(&ltc));
                debug_assert!(!ctrl_ltc.contains(&ltc));
                debug_assert!(!prvt_ltc.contains(&ltc));
                debug_assert!(!intf_ltc.contains(&ltc));
                // check that indices are unique, and store them
                debug_assert!(extl_ltc.insert(ltc));
            }

            // check that extl, intf, and prvt contain obs, ctrl, and wire
            for (ltc, _) in ctrl[0].iter() {
                debug_assert!(intf_ltc.contains(&ltc) || prvt_ltc.contains(&ltc));
            }
            for (ltc, _) in obs[0].iter() {
                debug_assert!(extl_ltc.contains(&ltc) || intf_ltc.contains(&ltc));
            }
            for (ltc, _) in wire[0].iter() {
                debug_assert!(
                    extl_ltc.contains(&ltc) || intf_ltc.contains(&ltc) || prvt_ltc.contains(&ltc)
                );
            }

            let nxt_to_dtype_get = |w| nxt_to_ltc.get(&w).and_then(|z| ltc_to_dtype.get(z));

            // check atoms consistency
            let mut written: HashSet<usize> = HashSet::from_iter(extl[1].iter().map(|(w, _)| w));
            for atom in atoms.iter() {
                for (ltc, dtype) in atom.read.iter() {
                    // reads are latched, and dtype matches
                    debug_assert_eq!(Some(&dtype), ltc_to_dtype.get(&ltc));
                }
                for (ltc, dtype) in atom.wait.iter() {
                    // awaits are next, and dtype matches
                    debug_assert_eq!(Some(&dtype), nxt_to_dtype_get(ltc));
                    // await order is consistent
                    debug_assert!(written.contains(&ltc));
                }
                for (nxt, dtype) in atom.ctrl.iter() {
                    // controls are next, and dtype matches
                    debug_assert_eq!(Some(&dtype), nxt_to_dtype_get(nxt));
                    // controls are disjoint
                    debug_assert!(written.insert(nxt));
                }
            }

            // check that all module control wires are written/controlled by an atom
            for (nxt, _) in ctrl[1].iter() {
                debug_assert!(written.contains(&nxt));
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

    pub fn new<A>(wire: [Wire<D>; 2], atoms: A) -> Result<Self, &'static str>
    where
        A: IntoIterator<Item = Atom<D, I>> + Sized,
    {
        // Check and store wire index + dtype information
        if wire[0].len() != wire[1].len() {
            return Err("len mismatch in latched and next wires");
        }
        let mut ltc_to_dtype: HashMap<usize, &D> = HashMap::new();
        let mut nxt_to_ltc: HashMap<usize, usize> = HashMap::new();
        for ((ltc, dtype), (nxt, ntype)) in wire[0].iter().zip(wire[1].iter()) {
            if dtype != ntype {
                return Err("dtype mismatch in latched and next wires");
            }
            if ltc_to_dtype.insert(ltc, dtype).is_some() {
                return Err("duplicate latched wire");
            }
            if nxt_to_ltc.insert(nxt, ltc).is_some() {
                return Err("duplicate next wire");
            }
        }

        // Check atoms consistency and infer control wires
        let mut ctrl_nxt: HashSet<usize> = HashSet::new();
        let atoms_iter = atoms.into_iter();
        let mut past_atoms: Vec<Atom<D, I>> = Vec::with_capacity(atoms_iter.size_hint().0);
        for atom in atoms_iter {
            for (ltc, dtype) in atom.read.iter() {
                if ltc_to_dtype.get(&ltc) != Some(&dtype) {
                    return Err("atom read not latched or dtype mismatch");
                }
            }
            for (nxt, dtype) in atom.wait.iter() {
                if nxt_to_ltc.get(&nxt).and_then(|i| ltc_to_dtype.get(i)) != Some(&dtype) {
                    return Err("atom await not next or dtype mismatch");
                }
            }
            for (nxt, dtype) in atom.ctrl.iter() {
                if nxt_to_ltc.get(&nxt).and_then(|i| ltc_to_dtype.get(i)) != Some(&dtype) {
                    return Err("atom control not next or dtype mismatch");
                }
                if !ctrl_nxt.insert(nxt) {
                    return Err("shared or duplicate atom control wire");
                }
            }

            // TODO check that temporaries are decoupled from module wires

            for past_atom in past_atoms.iter() {
                if past_atom.awaits(&atom) {
                    return Err("inconsistent awaiting order");
                }
            }
            past_atoms.push(atom);
        }

        // Build ctrl and extl wires based on inferred control set
        let mut ctrl_0: Vec<(usize, D)> = Vec::with_capacity(ctrl_nxt.len());
        let mut ctrl_1: Vec<(usize, D)> = Vec::with_capacity(ctrl_nxt.len());
        let mut extl_0: Vec<(usize, D)> = Vec::with_capacity(wire[0].len() - ctrl_nxt.len());
        let mut extl_1: Vec<(usize, D)> = Vec::with_capacity(wire[0].len() - ctrl_nxt.len());

        for (nxt, dtype) in wire[1].iter() {
            if ctrl_nxt.contains(&nxt) {
                ctrl_0.push((*nxt_to_ltc.get(&nxt).unwrap(), dtype.clone()));
                ctrl_1.push((nxt, dtype.clone()));
            } else {
                extl_0.push((*nxt_to_ltc.get(&nxt).unwrap(), dtype.clone()));
                extl_1.push((nxt, dtype.clone()));
            }
        }

        // Build wire pairs
        let extl = [Wire::new_unchecked(extl_0), Wire::new_unchecked(extl_1)];
        let ctrl = [Wire::new_unchecked(ctrl_0), Wire::new_unchecked(ctrl_1)];
        let prvt = [Wire::none(), Wire::none()];
        let intf = ctrl.clone();
        let obs = wire.clone();

        Ok(Self::new_unchecked(
            extl, intf, prvt, ctrl, obs, wire, past_atoms,
        ))
    }

    pub fn sequential<V, U>(wire: [Wire<D>; 2], init: V, update: U) -> Result<Self, &'static str>
    where
        V: IntoIterator<Item = Term<D, I>>,
        U: IntoIterator<Item = Term<D, I>>,
    {
        let atom = Atom::with_module_wire(&wire, Vec::from_iter(init), Vec::from_iter(update))?;
        Self::new(wire, [atom])
    }
}

impl<D: fmt::Display, I: fmt::Display> Module<D, I> {
    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, pad: &str) -> fmt::Result {
        const BOLD: &str = "\x1b[1m";
        const RESET: &str = "\x1b[0m";
        const INDENT: &str = "  ";
        const INDENT2: &str = "    ";

        writeln!(f, "{pad}{BOLD}module{RESET}")?;
        if !self.extl[0].is_empty() {
            writeln!(f, "{pad}{INDENT}{BOLD}external{RESET}")?;
        }
        for ((ltc, _), (nxt, dtype)) in self.extl[0].iter().zip(self.extl[1].iter()) {
            writeln!(f, "{pad}{INDENT2}w{ltc}, w{nxt} : {dtype}")?;
        }
        if !self.intf[0].is_empty() {
            writeln!(f, "{pad}{INDENT}{BOLD}interface{RESET}")?;
        }
        for ((ltc, _), (nxt, dtype)) in self.intf[0].iter().zip(self.intf[1].iter()) {
            writeln!(f, "{pad}{INDENT2}w{ltc}, w{nxt} : {dtype}")?;
        }
        if !self.prvt[0].is_empty() {
            writeln!(f, "{pad}{INDENT}{BOLD}private{RESET}")?;
        }
        for ((ltc, _), (nxt, dtype)) in self.prvt[0].iter().zip(self.prvt[1].iter()) {
            writeln!(f, "{pad}{INDENT2}w{ltc}, w{nxt} : {dtype}")?;
        }
        for atom in self.atoms.iter() {
            atom.fmt_indent(f, &format!("{pad}{INDENT}"))?;
        }
        Ok(())
    }
}

impl<D: fmt::Display, I: fmt::Display> fmt::Display for Module<D, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, "")
    }
}
