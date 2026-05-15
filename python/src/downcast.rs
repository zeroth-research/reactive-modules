use crate::types::{DType, IType};
use std::collections::HashMap;

/// cast DType to LIA Type
impl TryFrom<&DType> for theory::lia::Type {
    type Error = String;

    fn try_from(d: &DType) -> Result<Self, Self::Error> {
        match d {
            DType::Int(shape) if shape.len() == 2 => Ok(theory::lia::Type::Int(shape[0], shape[1])),
            DType::Bool(shape) if shape.len() == 2 => {
                Ok(theory::lia::Type::Bool(shape[0], shape[1]))
            }
            t => Err(format!("{} cannot be converted to lia::Type", t)),
        }
    }
}

/// cast DType to RLA Type
impl TryFrom<&DType> for theory::rla::Type {
    type Error = String;

    fn try_from(d: &DType) -> Result<Self, Self::Error> {
        match d {
            DType::Real(shape) if shape.len() == 2 => {
                Ok(theory::rla::Type::Real(shape[0], shape[1]))
            }
            DType::Bool(shape) if shape.len() == 2 => {
                Ok(theory::rla::Type::Bool(shape[0], shape[1]))
            }
            t => Err(format!("{} cannot be converted to rla::Type", t)),
        }
    }
}

/// cast DType to BV Type
impl TryFrom<&DType> for theory::bv::Type {
    type Error = String;

    fn try_from(d: &DType) -> Result<Self, Self::Error> {
        match d {
            DType::UWord(bw) => Ok(theory::bv::Type::U(*bw as usize, 1, 1)),
            DType::SWord(bw) => Ok(theory::bv::Type::S(*bw as usize, 1, 1)),
            DType::Bool(shape) if shape.len() == 2 => {
                Ok(theory::bv::Type::U(1, shape[0], shape[1]))
            }
            t => Err(format!("{} cannot be converted to bv::Type", t)),
        }
    }
}

/// cast IType to BV operations
impl TryFrom<&IType> for theory::bv::BV {
    type Error = String;

    fn try_from(op: &IType) -> Result<Self, Self::Error> {
        use theory::bv::BV;
        Ok(match op {
            IType::Add() => BV::Add,
            IType::Mul() => BV::Mul,
            IType::MatMul() => BV::MatMul,
            IType::And() => BV::And,
            IType::Or() => BV::Or,
            IType::Xor() => BV::Xor,
            IType::Not() => BV::Not,
            IType::Le() => BV::Le,
            IType::Lt() => BV::Lt,
            IType::Ge() => BV::Ge,
            IType::Gt() => BV::Gt,
            IType::Eq() => BV::Eq,
            IType::Neq() => BV::Ne,
            IType::Ite() => BV::Ite,
            IType::Id() => BV::Id,
            // TODO: we have to have ConstS and ConstU for constants of different type.
            IType::ConstInt(v) => BV::Const(vec![vec![*v as usize]]),
            t => return Err(format!("Cannot convert {} to BV operation", t)),
        })
    }
}

/// cast IType to LIA operations
impl TryFrom<&IType> for theory::lia::LIA {
    type Error = String;

    fn try_from(op: &IType) -> Result<Self, Self::Error> {
        use theory::lia::LIA;
        Ok(match op {
            IType::Add() => LIA::Add,
            IType::Eq() => LIA::Eq,
            IType::Neq() => LIA::Ne,
            IType::Lt() => LIA::Lt,
            IType::Le() => LIA::Le,
            IType::Gt() => LIA::Gt,
            IType::Ge() => LIA::Ge,
            IType::And() => LIA::And,
            IType::Or() => LIA::Or,
            IType::Not() => LIA::Not,
            IType::Xor() => LIA::Xor,
            IType::Ite() => LIA::Ite,
            IType::Id() => LIA::Id,
            IType::Argmax() => LIA::Argmax,
            IType::ReLU() => LIA::ReLU,
            IType::ConstInt(v) => LIA::ConstInt(vec![vec![*v]]),
            IType::ConstBool(b) => LIA::ConstBool(vec![vec![*b]]),
            t => return Err(format!("Cannot convert {} to LIA operation", t)),
        })
    }
}

/// cast IType to RLA operations
impl TryFrom<&IType> for theory::rla::RLA {
    type Error = String;

    fn try_from(op: &IType) -> Result<Self, Self::Error> {
        use theory::rla::RLA;
        Ok(match op {
            IType::Add() => RLA::Add,
            IType::Eq() => RLA::Eq,
            IType::Neq() => RLA::Ne,
            IType::Lt() => RLA::Lt,
            IType::Le() => RLA::Le,
            IType::Gt() => RLA::Gt,
            IType::Ge() => RLA::Ge,
            IType::And() => RLA::And,
            IType::Or() => RLA::Or,
            IType::Not() => RLA::Not,
            IType::Xor() => RLA::Xor,
            IType::Ite() => RLA::Ite,
            IType::Id() => RLA::Id,
            IType::Argmax() => RLA::Argmax,
            IType::ReLU() => RLA::ReLU,
            IType::ConstInt(v) => RLA::ConstReal(vec![vec![*v as f64]]),
            IType::ConstBool(b) => RLA::ConstBool(vec![vec![*b]]),
            t => return Err(format!("Cannot convert {} to RLA operation", t)),
        })
    }
}

/// Translate `term` to theory `U` using TryInto<IType>.
/// Param `map` keeps mapping of IDs from old wires to new wires (new terms will have new wires
/// with different dtypes, so we have to map those).
fn remap_term<U>(
    term: &base::Term<IType>,
    map: &HashMap<usize, base::Wire<U::DType>>,
) -> Result<base::Term<U>, String>
where
    U: theory::Theory,
    U::DType: Eq + Clone,
    for<'a> &'a IType: TryInto<U, Error = String>,
{
    let new_itype = term.itype().try_into()?;
    let new_write: Vec<base::Wire<U::DType>> =
        term.write().wires().map(|w| map[&w.id()].clone()).collect();
    let new_read: Vec<base::Wire<U::DType>> =
        term.read().wires().map(|w| map[&w.id()].clone()).collect();
    base::Term::function(new_itype, new_write, new_read)
}

/// Translate `atom` to theory `U` using `TryInto` for operations and `map` for rewiring.
fn remap_atom<U>(
    atom: &base::Atom<IType>,
    map: &HashMap<usize, base::Wire<U::DType>>,
) -> Result<base::Atom<U>, String>
where
    U: theory::Theory,
    U::DType: Eq + Clone,
    for<'a> &'a IType: TryInto<U, Error = String>,
{
    let new_latched: Vec<base::Wire<U::DType>> =
        atom.read().wires().map(|w| map[&w.id()].clone()).collect();
    let new_next: Vec<base::Wire<U::DType>> = atom
        .ctrl()
        .wires()
        .chain(atom.wait().wires())
        .map(|w| map[&w.id()].clone())
        .collect();
    let new_init: Vec<base::Term<U>> = atom
        .init()
        .iter()
        .map(|t| remap_term(t, map))
        .collect::<Result<_, _>>()?;
    let new_update: Vec<base::Term<U>> = atom
        .update()
        .iter()
        .map(|t| remap_term(t, map))
        .collect::<Result<_, _>>()?;
    Ok(base::Atom::sequential(
        new_latched.iter(),
        new_next.iter(),
        new_init,
        new_update,
    )?)
}

pub fn downcast_module<U>(module: &base::Module<IType>) -> Result<base::Module<U>, String>
where
    U: theory::Theory,
    U::DType: Eq + Clone + std::fmt::Debug,
    for<'a> &'a DType: TryInto<U::DType, Error = String>,
    for<'a> &'a IType: TryInto<U, Error = String>,
{
    // re-map wires from old DType to new DType
    // XXX: we could do this lazily to save some work if downcasting fails,
    // but let's start with simple solution, we can optimize later
    let mut map: HashMap<usize, base::Wire<U::DType>> = HashMap::new();
    for wire in module
        .extl()
        .wires()
        .chain(module.intf().wires())
        .chain(module.prvt().wires())
        .chain(module.temp())
    {
        let new_dtype = wire.dtype().try_into()?;
        map.insert(wire.id(), base::Wire::new(new_dtype));
    }

    let obs: Vec<[base::Wire<U::DType>; 2]> = module
        .obs()
        .iter()
        .map(|[ltc, nxt]| [map[&ltc.id()].clone(), map[&nxt.id()].clone()])
        .collect();

    let prvt: Vec<[base::Wire<U::DType>; 2]> = module
        .prvt()
        .iter()
        .map(|[ltc, nxt]| [map[&ltc.id()].clone(), map[&nxt.id()].clone()])
        .collect();

    let atoms: Vec<base::Atom<U>> = module
        .atoms()
        .iter()
        .map(|a| remap_atom(a, &map))
        .collect::<Result<_, _>>()?;

    Ok(base::Module::partially_observable(obs, prvt, atoms)?)
}

pub fn downcast_module_to_bv(
    module: &base::Module<IType>,
) -> Result<base::Module<theory::bv::BV>, String> {
    downcast_module(module)
}

pub fn downcast_module_to_lia(
    module: &base::Module<IType>,
) -> Result<base::Module<theory::lia::LIA>, String> {
    downcast_module(module)
}

pub fn downcast_module_to_rla(
    module: &base::Module<IType>,
) -> Result<base::Module<theory::rla::RLA>, String> {
    downcast_module(module)
}
