use crate::primitives::action::Action;
use crate::primitives::atom::Atom;
use crate::primitives::variable::{VarIterExt, Variable};
use std::collections::HashSet;

pub struct Module<V: Variable, S, I: Action<V, S>, U: Action<V, S>>
where
    for<'a> &'a S: IntoIterator<Item = &'a V>,
    for<'a> <&'a S as IntoIterator>::IntoIter: ExactSizeIterator,
{
    /// external
    pub extl: S,
    /// interface
    pub intf: S,
    /// private
    pub prvt: S,
    /// atoms
    pub atoms: Vec<Atom<V, S, I, U>>,
}

impl<V: Variable, S, I: Action<V, S>, U: Action<V, S>> Module<V, S, I, U>
where
    for<'a> &'a S: IntoIterator<Item = &'a V>,
    for<'a> <&'a S as IntoIterator>::IntoIter: ExactSizeIterator,
{
    pub fn new(extl: S, intf: S, prvt: S, atoms: Vec<Atom<V, S, I, U>>) -> Self {
        #[cfg(debug_assertions)]
        {
            debug_assert!(!extl.has_duplicates() && extl.is_latched()); // TODO: SDS is this trait automatically available once the crate loads?
            debug_assert!(!intf.has_duplicates() && intf.is_latched());
            debug_assert!(!prvt.has_duplicates() && prvt.is_latched());
            debug_assert!(VarIterExt::pairwise_disjoint(&[extl, intf, prvt]));

            // Check that the control variables of all atoms
            // are contained in the module.
            let mut ctr_atoms = HashSet::<&V>::new();
            for atom in atoms.iter() {
                ctr_atoms.extend(&atom.ctr);
            }
            let mut ctr_module = HashSet::new();
            ctr_module.extend(&intf);
            ctr_module.extend(&prvt);
            debug_assert_eq!(ctr_atoms, ctr_module);

            // Check that the control variables of all atoms are disjoint.
            for (i, atom1) in atoms.iter().enumerate() {
                for atom2 in atoms.iter().skip(i + 1) {
                    debug_assert!(atom1.ctr.as_set().is_disjoint(&atom2.ctr.as_set()))
                }
            }

            // Check that atoms are in topological order.
            for (i, atom1) in atoms.iter().enumerate() {
                for atom2 in atoms.iter().skip(i + 1) {
                    debug_assert!(!atom1.awaits(atom2))
                }
            }
        }

        Self {
            extl,
            intf,
            prvt,
            atoms,
        }
    }

    pub fn vars_iter(&self) -> impl Iterator<Item = &V> {
        self.extl
            .into_iter()
            .chain(self.intf.into_iter())
            .chain(self.prvt.into_iter())
    }

    pub fn ctr_iter(&self) -> impl Iterator<Item = &V> {
        self.intf.into_iter().chain(self.prvt.into_iter())
    }
}
