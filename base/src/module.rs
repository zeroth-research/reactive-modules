use crate::atom::Atom;
use crate::wire::Wire;
use crate::write_indented;
use std::fmt;

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
impl<D: Clone + Eq, I> Module<D, I> {
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
        let offset: isize = (wire[1].ranges[0].start as isize) - (wire[0].ranges[0].start as isize);
        let extl = [extl_1.twin(offset).unwrap(), extl_1]; // unwrap cuz checks above ensure
        let obs = [obs_1.twin(offset).unwrap(), obs_1]; // unwrap cuz checks above ensure
        let ctrl = [ctrl_1.twin(offset).unwrap(), ctrl_1]; // unwrap cuz checks above ensure
        let intf = ctrl.clone();
        let prvt = [Wire::<D>::none(), Wire::<D>::none()];

        Ok(Module {
            extl,
            intf,
            prvt,
            ctrl,
            obs,
            wire,
            atoms,
        })
    }
}

impl<D: fmt::Display, I: fmt::Display> fmt::Display for Module<D, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "module")?;
        writeln!(f, "  external ({} | {})", self.extl[0], self.extl[1])?;
        writeln!(f, "  interface ({} | {})", self.intf[0], self.intf[1])?;
        writeln!(f, "  private ({} | {})", self.prvt[0], self.prvt[1])?;
        for atom in self.atoms.iter() {
            write_indented(f, "  ", atom)?;
        }
        Ok(())
    }
}
