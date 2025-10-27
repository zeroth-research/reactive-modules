use crate::atom::Atom;
use crate::wire::Wire;

#[derive(Debug, Clone)]
pub struct WirePair<D> {
    pub latched: Wire<D>,
    pub next: Wire<D>,
}

impl<D: Eq> WirePair<D> {
    pub fn is_twin(&self) -> bool {
        self.latched.is_twin(&self.next)
    }
}

impl<D> WirePair<D> {
    pub fn empty() -> Self {
        WirePair {
            latched: Wire::empty(),
            next: Wire::empty(),
        }
    }
}

impl<D> From<(Wire<D>, Wire<D>)> for WirePair<D> {
    fn from(value: (Wire<D>, Wire<D>)) -> Self {
        WirePair {
            latched: value.0,
            next: value.1,
        }
    }
}

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
    extl: WirePair<D>,
    intf: WirePair<D>,
    prvt: WirePair<D>,
    ctrl: WirePair<D>,
    obs: WirePair<D>,
    wire: WirePair<D>,

    /// The atoms of this module.
    /// The atoms must be stored in a *consistent* linear order
    /// as defined in the reactive modules paper.
    atoms: Vec<Atom<D, I>>,
}
impl<D: Clone + Eq, I> Module<D, I> {
    #[allow(clippy::unwrap_used)]
    pub fn new(wire: WirePair<D>, atoms: Vec<Atom<D, I>>) -> Result<Self, &'static str> {
        // Check latched and next wires
        if !wire.is_twin() {
            return Err("latched and next wires are not matching");
        }

        // Infer next controlled wires from atoms
        let mut ctrl_1: Wire<D> = Wire::empty();
        for (i, atom) in atoms.iter().enumerate() {
            if !atom.read.is_subset(&wire.latched) {
                return Err("atom read is not latched");
            }
            if !atom.wait.is_subset(&wire.next) {
                return Err("atom wait is not next");
            }
            if !atom.ctrl.is_subset(&wire.next) {
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
        let extl_1 = wire.next.difference(&ctrl_1).unwrap(); // unwrap cuz checks above ensure
        let obs_1 = ctrl_1.union(&extl_1).unwrap(); // unwrap cuz checks above ensure

        // offset extl, ctrl, and obs backward to obtain latched wires
        let offset: isize =
            (wire.next.ranges[0].start as isize) - (wire.latched.ranges[0].start as isize);
        let extl = (extl_1.twin(offset).unwrap(), extl_1).into(); // unwrap cuz checks above ensure
        let obs = (obs_1.twin(offset).unwrap(), obs_1).into(); // unwrap cuz checks above ensure
        let ctrl: WirePair<D> = (ctrl_1.twin(offset).unwrap(), ctrl_1).into(); // unwrap cuz checks above ensure
        let intf = ctrl.clone();
        let prvt = WirePair::<D>::empty();

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
