// use crate::types::{DType, IType};
// use std::collections::HashMap;
//
// /// cast DType to LIA Type
// impl TryFrom<&DType> for theory::lia::Type {
//     type Error = String;
//
//     fn try_from(d: &DType) -> Result<Self, Self::Error> {
//         match &d.0 {
//             theory::any::Type::Int(shape) => Ok(theory::lia::Type::Int(*shape)),
//             theory::any::Type::Bool(shape) => Ok(theory::lia::Type::Bool(*shape)),
//             _ => Err(format!("{} cannot be converted to lia::Type", d)),
//         }
//     }
// }
//
// /// cast DType to LRA Type
// impl TryFrom<&DType> for theory::lra::Type {
//     type Error = String;
//
//     fn try_from(d: &DType) -> Result<Self, Self::Error> {
//         match &d.0 {
//             theory::any::Type::Real(shape) => Ok(theory::lra::Type::Real(*shape)),
//             theory::any::Type::Bool(shape) => Ok(theory::lra::Type::Bool(*shape)),
//             _ => Err(format!("{} cannot be converted to lra::Type", d)),
//         }
//     }
// }
//
// /// cast DType to BV Type
// impl TryFrom<&DType> for theory::bv::Type {
//     type Error = String;
//
//     fn try_from(d: &DType) -> Result<Self, Self::Error> {
//         match &d.0 {
//             theory::any::Type::BV(bw, shape) => Ok(theory::bv::Type::BV(*bw, *shape)),
//             theory::any::Type::Bool(shape) => Ok(theory::bv::Type::BV(1, *shape)),
//             _ => Err(format!("{} cannot be converted to bv::Type", d)),
//         }
//     }
// }
//
// /// cast IType to BV operations
// impl TryFrom<&IType> for theory::bv::BV {
//     type Error = String;
//
//     fn try_from(op: &IType) -> Result<Self, Self::Error> {
//         match &op.0 {
//             theory::any::Any::BV(bv) => Ok(bv.clone()),
//             _ => Err(format!("Cannot convert {} to BV operation", op)),
//         }
//     }
// }
//
// /// cast IType to LIA operations
// impl TryFrom<&IType> for theory::lia::LIA {
//     type Error = String;
//
//     fn try_from(op: &IType) -> Result<Self, Self::Error> {
//         match &op.0 {
//             theory::any::Any::LIA(lia) => Ok(lia.clone()),
//             _ => Err(format!("Cannot convert {} to LIA operation", op)),
//         }
//     }
// }
//
// /// cast IType to LRA operations
// impl TryFrom<&IType> for theory::lra::LRA {
//     type Error = String;
//
//     fn try_from(op: &IType) -> Result<Self, Self::Error> {
//         match &op.0 {
//             theory::any::Any::LRA(lra) => Ok(lra.clone()),
//             _ => Err(format!("Cannot convert {} to LRA operation", op)),
//         }
//     }
// }
//
// /// Translate `term` to theory `U` using TryInto<IType>.
// /// Param `map` keeps mapping of IDs from old wires to new wires (new terms will have new wires
// /// with different dtypes, so we have to map those).
// fn remap_term<U>(
//     term: &base::Term<IType>,
//     map: &HashMap<usize, base::Wire<U::DType>>,
// ) -> Result<base::Term<U>, String>
// where
//     U: theory::Theory,
//     U::DType: Eq + Clone + std::fmt::Display,
//     for<'a> &'a IType: TryInto<U, Error = String>,
// {
//     let new_itype = term.itype().try_into()?;
//     let new_write: Vec<base::Wire<U::DType>> =
//         term.write().wires().map(|w| map[&w.id()].clone()).collect();
//     let new_read: Vec<base::Wire<U::DType>> =
//         term.read().wires().map(|w| map[&w.id()].clone()).collect();
//     base::Term::function(new_itype, new_write, new_read)
// }
//
// /// Translate `atom` to theory `U` using `TryInto` for operations and `map` for rewiring.
// fn remap_atom<U>(
//     atom: &base::Atom<IType>,
//     map: &HashMap<usize, base::Wire<U::DType>>,
// ) -> Result<base::Atom<U>, String>
// where
//     U: theory::Theory,
//     U::DType: Eq + Clone + std::fmt::Display,
//     for<'a> &'a IType: TryInto<U, Error = String>,
// {
//     let new_latched: Vec<base::Wire<U::DType>> =
//         atom.read().wires().map(|w| map[&w.id()].clone()).collect();
//     let new_next: Vec<base::Wire<U::DType>> = atom
//         .ctrl()
//         .wires()
//         .chain(atom.wait().wires())
//         .map(|w| map[&w.id()].clone())
//         .collect();
//     let new_init: Vec<base::Term<U>> = atom
//         .init()
//         .iter()
//         .map(|t| remap_term(t, map))
//         .collect::<Result<_, _>>()?;
//     let new_update: Vec<base::Term<U>> = atom
//         .update()
//         .iter()
//         .map(|t| remap_term(t, map))
//         .collect::<Result<_, _>>()?;
//     Ok(base::Atom::sequential(
//         new_latched.iter(),
//         new_next.iter(),
//         new_init,
//         new_update,
//     )?)
// }
//
// pub fn downcast_module<U>(module: &base::Module<IType>) -> Result<base::Module<U>, String>
// where
//     U: theory::Theory,
//     U::DType: Eq + Clone + std::fmt::Debug + std::fmt::Display,
//     for<'a> &'a DType: TryInto<U::DType, Error = String>,
//     for<'a> &'a IType: TryInto<U, Error = String>,
// {
//     // re-map wires from old DType to new DType
//     let mut map: HashMap<usize, base::Wire<U::DType>> = HashMap::new();
//     for wire in module
//         .extl()
//         .wires()
//         .chain(module.intf().wires())
//         .chain(module.prvt().wires())
//         .chain(module.temp())
//     {
//         let new_dtype = wire.dtype().try_into()?;
//         map.insert(wire.id(), base::Wire::new(new_dtype));
//     }
//
//     let obs: Vec<[base::Wire<U::DType>; 2]> = module
//         .obs()
//         .iter()
//         .map(|[ltc, nxt]| [map[&ltc.id()].clone(), map[&nxt.id()].clone()])
//         .collect();
//
//     let prvt: Vec<[base::Wire<U::DType>; 2]> = module
//         .prvt()
//         .iter()
//         .map(|[ltc, nxt]| [map[&ltc.id()].clone(), map[&nxt.id()].clone()])
//         .collect();
//
//     let atoms: Vec<base::Atom<U>> = module
//         .atoms()
//         .iter()
//         .map(|a| remap_atom(a, &map))
//         .collect::<Result<_, _>>()?;
//
//     Ok(base::Module::partially_observable(obs, prvt, atoms)?)
// }
//
// pub fn downcast_module_to_bv(
//     module: &base::Module<IType>,
// ) -> Result<base::Module<theory::bv::BV>, String> {
//     downcast_module(module)
// }
//
// pub fn downcast_module_to_lia(
//     module: &base::Module<IType>,
// ) -> Result<base::Module<theory::lia::LIA>, String> {
//     downcast_module(module)
// }
//
// pub fn downcast_module_to_lra(
//     module: &base::Module<IType>,
// ) -> Result<base::Module<theory::lra::LRA>, String> {
//     downcast_module(module)
// }
