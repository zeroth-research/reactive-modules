use base::wire::{Interface, Wire};
use smt::dtype::DType as DTypeSMT;
use smt::itype::IType as ITypeSMT;
use std::cmp::max;
use std::collections::HashMap;
use std::convert::TryInto;

use crate::conversions::ModuleConverter;
use crate::itype::{ArithOp, CmpOp, LogicalOp};
use crate::val::Val;
use crate::{DType, IType, ToyAtom, ToyTerm};

type SmtModule = base::Module<DTypeSMT, ITypeSMT>;
type SmtAtom = base::Atom<DTypeSMT, ITypeSMT>;
type SmtTerm = base::Term<DTypeSMT, ITypeSMT>;

impl TryInto<DTypeSMT> for &DType {
    type Error = &'static str;

    fn try_into(self) -> Result<DTypeSMT, Self::Error> {
        match self {
            DType::Int => Ok(DTypeSMT::Int),
            DType::Bool => Ok(DTypeSMT::Bool),
            DType::Real => Ok(DTypeSMT::Real),
            _ => Err("Cannot convert this type to smt::dtype::DType"),
        }
    }
}

impl TryInto<DTypeSMT> for DType {
    type Error = &'static str;

    fn try_into(self) -> Result<DTypeSMT, Self::Error> {
        (&self).try_into()
    }
}

impl TryInto<smt::itype::Val> for &Val {
    type Error = &'static str;

    fn try_into(self) -> Result<smt::itype::Val, Self::Error> {
        use smt::itype::Val as ValSMT;
        match self {
            Val::Int(v) => Ok(ValSMT::Int(*v)),
            Val::Bool(v) => Ok(ValSMT::Bool(*v)),
            Val::Real(v) => Ok(ValSMT::Real(*v)),

            _ => Err("Cannot convert this type to smt::dtype::DType"),
        }
    }
}

impl TryInto<smt::itype::Val> for Val {
    type Error = &'static str;

    fn try_into(self) -> Result<smt::itype::Val, Self::Error> {
        (&self).try_into()
    }
}

/// A helpre struct for translating a toy module into an smt module.
/// The translation must map atoms to atoms (one to one), terms to terms (possibly one to many
/// if a single term needs to be encoded into multiple smt terms) and wires
/// to wires (one to one, possibly adding new wires if there are new terms added).
struct SmtTranslator {
    // We use this map to keep mapping of our wires to wires in the translated module.
    // The values are ((new_id, new_type), orig_type).
    // We keep the original type to do checks if everything is correct.
    wires_mapping: HashMap<usize, ((usize, DTypeSMT), DType)>,

    // our ID counter for created wire IDs
    id_cnt: usize,
}

type Err = &'static str;

impl SmtTranslator {
    fn new() -> Self {
        SmtTranslator {
            wires_mapping: HashMap::new(),
            // assign our id counter something different than 0,
            // so that it is easier to catch bugs. We'll can go with 0
            // when the implementation is done and working.
            // (Then, we can actually avoid mapping most of the IDs)
            id_cnt: 1_000_000,
        }
    }

    /// Remap all variables of module so that they have ID greater by 1 000 000.
    /// This must be done before the translation itself as the function does not
    /// check for existing IDs (and it makes no sense to do that in the middle
    /// of translation). Note that calling this funciton is optional, the translation
    /// will work without it, only the IDs will not be easily eye-mappable to the original
    /// IDs.
    fn remap_variables(&mut self, module: &crate::ToyModule) -> Result<(), Err> {
        let mut max_id: usize = 0;
        const OFFSET: usize = 1_000_000;

        // process the module wires
        for [wl, wn] in module.extl().iter().chain(module.ctrl().iter()) {
            let new_id = wl.id() + OFFSET;
            let new_ty = (wl.dtype()).try_into()?;
            assert!(!self.wires_mapping.contains_key(&new_id));
            self.wires_mapping
                .insert(wl.id(), ((new_id, new_ty), *wl.dtype()));

            let new_id = wn.id() + OFFSET;
            let new_ty = (wn.dtype()).try_into()?;
            assert!(!self.wires_mapping.contains_key(&new_id));
            self.wires_mapping
                .insert(wn.id(), ((new_id, new_ty), *wn.dtype()));

            max_id = max(wl.id(), max_id);
            max_id = max(wn.id(), max_id);
        }

        // process temporary wires in atoms (other wires used in atoms are already
        // included in the module wires)
        for atom in module.atoms() {
            for w in atom.temp() {
                let new_id = w.id() + OFFSET;
                let new_ty = (w.dtype()).try_into()?;
                assert!(!self.wires_mapping.contains_key(&new_id));
                self.wires_mapping
                    .insert(w.id(), ((new_id, new_ty), *w.dtype()));

                max_id = max(w.id(), max_id);
            }
        }

        // assign new wires completely different IDs so that we can directly
        // see they are new
        // TODO: check overflows..
        self.id_cnt = 2 * (OFFSET + max_id);

        Ok(())
    }

    /// Get a pair of (mapped_id, smt_type) for toy's id and type. If the id has not been mapped
    /// yet, it is mapped.
    fn get_wire(&mut self, id: usize, ty: &DType) -> Result<(usize, DTypeSMT), Err> {
        if let Some((pair, known_ty)) = self.wires_mapping.get(&id) {
            debug_assert!(known_ty == ty);
            debug_assert!(pair.1 == (*ty).try_into().unwrap());
            Ok(*pair)
        } else {
            let new_id = self.map_id_(id);
            let new_ty = (*ty).try_into()?;
            self.wires_mapping.insert(id, ((new_id, new_ty), *ty));
            Ok((new_id, new_ty))
        }
    }

    /// Get a mapped ID, return None if this ID is not mapped
    fn get_mapped_id(&self, id: usize) -> Option<usize> {
        if let Some(val) = self.wires_mapping.get(&id) {
            return Some(val.0.0);
        }
        None
    }

    /// Get a mapped ID or a new fresh ID to use
    fn map_id_(&mut self, id: usize) -> usize {
        if let Some(val) = self.wires_mapping.get(&id) {
            val.0.0
        } else {
            // take next ID from the counter
            self.id_cnt += 1;
            assert!(!self.wires_mapping.contains_key(&self.id_cnt));
            self.id_cnt
        }
    }

    fn translate_term(&mut self, term: &ToyTerm) -> Result<Vec<SmtTerm>, Err> {
        let read: Vec<(usize, DTypeSMT)> = term
            .read()
            .wires()
            .map(|w| self.get_wire(w.id(), w.dtype()).unwrap())
            .collect();
        let write: Vec<(usize, DTypeSMT)> = term
            .write()
            .wires()
            .map(|w| self.get_wire(w.id(), w.dtype()).unwrap())
            .collect();

        match term.itype() {
            IType::Id => {
                debug_assert!(read.len() == 1);
                debug_assert!(write.len() == 1);
                Ok(vec![SmtTerm::function(ITypeSMT::Id, write, read)?])
            }
            IType::Const(val) => {
                debug_assert!(read.is_empty());
                debug_assert!(write.len() == 1);
                Ok(vec![SmtTerm::function(
                    ITypeSMT::Num(val.try_into()?),
                    write,
                    read,
                )?])
            }
            IType::Cmp(op) => {
                debug_assert!(read.len() == 2);
                debug_assert!(write.len() == 1);
                match op {
                    CmpOp::Lt => Ok(vec![SmtTerm::function(
                        ITypeSMT::Cmp(smt::itype::CmpOp::Lt),
                        write,
                        read,
                    )?]),
                    CmpOp::Le => Ok(vec![SmtTerm::function(
                        ITypeSMT::Cmp(smt::itype::CmpOp::Le),
                        write,
                        read,
                    )?]),
                    CmpOp::Eq => Ok(vec![SmtTerm::function(
                        ITypeSMT::Cmp(smt::itype::CmpOp::Eq),
                        write,
                        read,
                    )?]),
                }
            }
            IType::Logical(op) => {
                debug_assert!(write.len() == 1);
                match op {
                    LogicalOp::And => {
                        debug_assert!(read.len() == 2);
                        Ok(vec![SmtTerm::function(
                            ITypeSMT::Logical(smt::itype::LogicalOp::And),
                            write,
                            read,
                        )?])
                    }
                    LogicalOp::Or => {
                        debug_assert!(read.len() == 2);
                        Ok(vec![SmtTerm::function(
                            ITypeSMT::Logical(smt::itype::LogicalOp::Or),
                            write,
                            read,
                        )?])
                    }
                    LogicalOp::Not => {
                        debug_assert!(read.len() == 1);
                        Ok(vec![SmtTerm::function(
                            ITypeSMT::Logical(smt::itype::LogicalOp::Not),
                            write,
                            read,
                        )?])
                    }
                }
            }

            IType::Arith(op) => {
                debug_assert!(read.len() == 2);
                debug_assert!(write.len() == 1);
                match op {
                    ArithOp::Add => Ok(vec![SmtTerm::function(
                        ITypeSMT::Arith(smt::itype::ArithOp::Add),
                        write,
                        read,
                    )?]),
                    ArithOp::Sub => Ok(vec![SmtTerm::function(
                        ITypeSMT::Arith(smt::itype::ArithOp::Sub),
                        write,
                        read,
                    )?]),
                    ArithOp::Mul => Ok(vec![SmtTerm::function(
                        ITypeSMT::Arith(smt::itype::ArithOp::Mul),
                        write,
                        read,
                    )?]),
                    ArithOp::Div => Ok(vec![SmtTerm::function(
                        ITypeSMT::Arith(smt::itype::ArithOp::Div),
                        write,
                        read,
                    )?]),
                }
            }
            IType::Ite => {
                debug_assert!(read.len() == 3);
                debug_assert!(write.len() == 1);
                Ok(vec![SmtTerm::function(ITypeSMT::Cond, write, read)?])
            }
            IType::IfThen => Err("Translating IfThen not implemented right now"),
            IType::Choose => Err("Translating Choose not implemented right now"),
        }
    }

    fn translate_atom(&mut self, atom: &ToyAtom) -> Result<(Vec<SmtTerm>, Vec<SmtTerm>), Err> {
        let mut init: Vec<SmtTerm> = Vec::new();
        let mut update: Vec<SmtTerm> = Vec::new();
        for term in atom.init() {
            init.extend(self.translate_term(term)?);
        }

        for term in atom.update() {
            update.extend(self.translate_term(term)?);
        }

        Ok((init, update))
    }

    fn translate_variables(
        &mut self,
        variables: &Interface<DType, 2>,
    ) -> Result<Interface<DTypeSMT, 2>, Err> {
        let mut latched: Vec<Wire<DTypeSMT>> = Vec::new();
        let mut nxt: Vec<Wire<DTypeSMT>> = Vec::new();
        for w in variables.iter() {
            latched.push(Wire::new(self.map_id_(w[0].id()), w[0].dtype().try_into()?));
            nxt.push(Wire::new(self.map_id_(w[1].id()), w[1].dtype().try_into()?));
        }

        Interface::try_from_iter(latched.into_iter().zip(nxt).map(|(w1, w2)| [w1, w2]))
    }
}

#[cfg(debug_assertions)]
fn check_variables(
    translator: &SmtTranslator,
    orig_vars: &Interface<DType, 2>,
    new_vars: &Interface<DTypeSMT, 2>,
) {
    for (w1, w2) in new_vars.wires().zip(orig_vars.wires()) {
        // check that the mapped ID matches
        if let Some(id) = translator.get_mapped_id(w2.id()) {
            debug_assert!(id == w1.id());
        }
        // check that the mapped type matches
        let ty: DTypeSMT = w2.dtype().try_into().unwrap();
        debug_assert!(ty == *w1.dtype());
    }
}

impl<'a> TryInto<SmtModule> for ModuleConverter<'a> {
    type Error = Err;

    fn try_into(self) -> Result<SmtModule, Self::Error> {
        // translate atoms, variables will be then derived automatically
        let mut translator = SmtTranslator::new();

        // this step is optional: we map variables such that they are shifted by 1000000 compared
        // to the variables of the Toy module. Using this offset helps finding bugs
        // in the translation since the IDs will be likely different from those in Toy module
        // (where they start from 0), second, we still can see quite easily what was the original ID.
        translator.remap_variables(self.0)?;

        let mut atom_funs: Vec<(Vec<SmtTerm>, Vec<SmtTerm>)> = Vec::new();
        for atom in self.0.atoms() {
            let (init, update) = translator.translate_atom(atom)?;
            atom_funs.push((init, update));
        }

        // we have to translate wires only after translating atoms, because now we have the mapping
        // for all the wires
        let extl = translator.translate_variables(self.0.extl())?;
        let intf = translator.translate_variables(self.0.intf())?;
        let prvt = translator.translate_variables(self.0.prvt())?;
        let obs = translator.translate_variables(self.0.obs())?;
        let ctrl = translator.translate_variables(self.0.ctrl())?;

        #[cfg(debug_assertions)]
        {
            for i in 0..=1 {
                debug_assert!(extl[i].len() == self.0.extl()[i].len());
                debug_assert!(intf[i].len() == self.0.intf()[i].len());
                debug_assert!(prvt[i].len() == self.0.prvt()[i].len());
                debug_assert!(obs[i].len() == self.0.obs()[i].len());
                debug_assert!(ctrl[i].len() == self.0.ctrl()[i].len());
            }

            check_variables(&translator, self.0.extl(), &extl);
            check_variables(&translator, self.0.intf(), &intf);
            check_variables(&translator, self.0.prvt(), &prvt);
            check_variables(&translator, self.0.obs(), &obs);
            check_variables(&translator, self.0.ctrl(), &ctrl);
        }

        let wire: Vec<[&Wire<DTypeSMT>; 2]> = obs.iter().chain(prvt.iter()).collect();

        // TODO: to private, add extra wires created during translation
        // (now there are none)

        let mut atoms: Vec<SmtAtom> = Vec::new();
        for (init, update) in atom_funs.into_iter() {
            let latched = wire.iter().map(|[w1, _]| *w1);
            let next = wire.iter().map(|[_, w2]| *w2);
            atoms.push(SmtAtom::sequential(latched, next, init, update)?);
        }

        SmtModule::partially_observable(obs, prvt, atoms)
    }
}
