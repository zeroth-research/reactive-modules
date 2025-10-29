use crate::atom::Atom;
use crate::wire::Wire;
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
            debug_assert!(Wire::is_twin(&extl[0], &extl[1]));
            debug_assert!(Wire::is_twin(&intf[0], &intf[1]));
            debug_assert!(Wire::is_twin(&prvt[0], &prvt[1]));
            debug_assert!(Wire::is_twin(&ctrl[0], &ctrl[1]));
            debug_assert!(Wire::is_twin(&obs[0], &obs[1]));
            debug_assert!(Wire::is_twin(&wire[0], &wire[1]));

            debug_assert!(Wire::is_disjoint(&extl[0], &intf[0]));
            debug_assert!(Wire::is_disjoint(&extl[0], &prvt[0]));
            debug_assert!(Wire::is_disjoint(&intf[0], &prvt[0]));

            debug_assert!(Wire::is_disjoint(&extl[1], &intf[1]));
            debug_assert!(Wire::is_disjoint(&extl[1], &prvt[1]));
            debug_assert!(Wire::is_disjoint(&intf[1], &prvt[1]));

            debug_assert_eq!(Wire::union(&extl[0], &intf[0]).unwrap(), obs[0]);
            debug_assert_eq!(Wire::union(&intf[0], &prvt[0]).unwrap(), ctrl[0]);

            debug_assert_eq!(Wire::union(&extl[1], &intf[1]).unwrap(), obs[1]);
            debug_assert_eq!(Wire::union(&intf[1], &prvt[1]).unwrap(), ctrl[1]);

            debug_assert_eq!(
                wire[0],
                extl[0].union(&intf[0]).unwrap().union(&prvt[0]).unwrap()
            );

            debug_assert_eq!(
                wire[1],
                extl[1].union(&intf[1]).unwrap().union(&prvt[1]).unwrap()
            );

            for (i, atom1) in atoms.iter().enumerate() {
                for atom2 in atoms.iter().skip(i + 1) {
                    // Check that the control variables of all atoms are pairwise disjoint.
                    debug_assert!(atom1.ctrl.is_disjoint(&atom2.ctrl));
                    // Check that atoms are in topological order.
                    debug_assert!(!atom1.awaits(atom2))
                }
            }
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

    #[allow(clippy::unwrap_used)]
    pub fn with_atoms(wire: [Wire<D>; 2], atoms: Vec<Atom<D, I>>) -> Result<Self, &'static str> {
        // Check latched and next wires
        if !wire[0].is_twin(&wire[1]) {
            return Err("latched and next wires are not matching");
        }

        // Infer next controlled wires from atoms
        let mut ctrl_1: Wire<D> = Wire::none();
        for (i, atom) in atoms.iter().enumerate() {
            if !atom.read.is_subset(&wire[0]) {
                return Err("atom read is not latched");
            }
            if !atom.wait.is_subset(&wire[1]) {
                return Err("atom wait is not next");
            }
            if !atom.ctrl.is_subset(&wire[1]) {
                return Err("atom ctrl is not next");
            }
            if !atom.ctrl.is_disjoint(&ctrl_1) {
                return Err("atoms are sharing controlled variables");
            }
            for past_atom in atoms.iter().take(i) {
                if past_atom.awaits(atom) {
                    return Err("inconsistent awaiting order");
                }
            }
            ctrl_1 = ctrl_1.union(&atom.ctrl).unwrap(); // unwrap cuz checks above ensure
        }

        // Infer next external and observable wires
        let extl_1 = wire[1].difference(&ctrl_1).unwrap(); // unwrap cuz checks above ensure
        let obs_1 = ctrl_1.union(&extl_1).unwrap(); // unwrap cuz checks above ensure

        // offset extl, ctrl, and obs backward to obtain latched wires
        let offset: isize = (wire[0].ranges[0].start as isize) - (wire[1].ranges[0].start as isize);
        let extl = [extl_1.twin(offset).unwrap(), extl_1]; // unwrap cuz checks above ensure
        let obs = [obs_1.twin(offset).unwrap(), obs_1]; // unwrap cuz checks above ensure
        let ctrl = [ctrl_1.twin(offset).unwrap(), ctrl_1]; // unwrap cuz checks above ensure
        let intf = ctrl.clone();
        let prvt = [Wire::<D>::none(), Wire::<D>::none()];

        Ok(Self::new_unchecked(
            extl, intf, prvt, ctrl, obs, wire, atoms,
        ))
    }
}
