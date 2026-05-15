use crate::types::{DType, IType};
use std::collections::HashMap;

impl TryFrom<&DType> for theory::lia::Type {
    type Error = ();

    fn try_from(d: &DType) -> Result<Self, ()> {
        match d {
            DType::Int(shape) if shape.len() == 2 => Ok(theory::lia::Type::Int(shape[0], shape[1])),
            DType::Bool(shape) if shape.len() == 2 => {
                Ok(theory::lia::Type::Bool(shape[0], shape[1]))
            }
            _ => Err(()),
        }
    }
}

impl TryFrom<&DType> for theory::rla::Type {
    type Error = ();

    fn try_from(d: &DType) -> Result<Self, ()> {
        match d {
            DType::Real(shape) if shape.len() == 2 => {
                Ok(theory::rla::Type::Real(shape[0], shape[1]))
            }
            DType::Bool(shape) if shape.len() == 2 => {
                Ok(theory::rla::Type::Bool(shape[0], shape[1]))
            }
            _ => Err(()),
        }
    }
}

impl TryFrom<&IType> for theory::lia::LIA {
    type Error = ();

    fn try_from(op: &IType) -> Result<Self, ()> {
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
            _ => return Err(()),
        })
    }
}

impl TryFrom<&IType> for theory::rla::RLA {
    type Error = ();

    fn try_from(op: &IType) -> Result<Self, ()> {
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
            _ => return Err(()),
        })
    }
}

fn remap_term<U, E>(
    term: &base::Term<IType>,
    f_itype: &impl Fn(&IType) -> Result<U, E>,
    map: &HashMap<usize, base::Wire<U::DType>>,
) -> Result<base::Term<U>, E>
where
    U: theory::Theory,
    U::DType: Eq + Clone,
{
    let new_itype = f_itype(term.itype())?;
    let new_write: Vec<base::Wire<U::DType>> =
        term.write().wires().map(|w| map[&w.id()].clone()).collect();
    let new_read: Vec<base::Wire<U::DType>> =
        term.read().wires().map(|w| map[&w.id()].clone()).collect();
    let new_write = base::Interface::unique(new_write).expect("remapped write is unique");
    let new_read = base::Interface::sequence(new_read).expect("remapped read is well-typed");
    Ok(base::Term::new_unchecked(new_itype, new_write, new_read))
}

fn remap_atom<U, E>(
    atom: &base::Atom<IType>,
    f_itype: &impl Fn(&IType) -> Result<U, E>,
    map: &HashMap<usize, base::Wire<U::DType>>,
) -> Result<base::Atom<U>, E>
where
    U: theory::Theory,
    U::DType: Eq + Clone,
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
        .map(|t| remap_term(t, f_itype, map))
        .collect::<Result<_, _>>()?;
    let new_update: Vec<base::Term<U>> = atom
        .update()
        .iter()
        .map(|t| remap_term(t, f_itype, map))
        .collect::<Result<_, _>>()?;
    let new_atom =
        base::Atom::sequential(new_latched.iter(), new_next.iter(), new_init, new_update)
            .expect("atom reconstruction invariants hold for a valid source module");
    Ok(new_atom)
}

pub fn downcast_module<U, E>(
    module: &base::Module<IType>,
    f_dtype: impl Fn(&DType) -> Result<U::DType, E>,
    f_itype: impl Fn(&IType) -> Result<U, E>,
) -> Result<base::Module<U>, E>
where
    U: theory::Theory,
    U::DType: Eq + Clone + std::fmt::Debug,
{
    let mut map: HashMap<usize, base::Wire<U::DType>> = HashMap::new();
    for wire in module
        .extl()
        .wires()
        .chain(module.intf().wires())
        .chain(module.prvt().wires())
        .chain(module.temp())
    {
        let new_dtype = f_dtype(wire.dtype())?;
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
        .map(|a| remap_atom(a, &f_itype, &map))
        .collect::<Result<_, _>>()?;

    let new_module = base::Module::partially_observable(obs, prvt, atoms)
        .expect("module reconstruction invariants hold for a valid source module");
    Ok(new_module)
}

pub fn downcast_module_to_lia(
    module: &base::Module<IType>,
) -> Result<base::Module<theory::lia::LIA>, ()> {
    downcast_module(
        module,
        |dtype| theory::lia::Type::try_from(dtype),
        |itype| theory::lia::LIA::try_from(itype),
    )
}

pub fn downcast_module_to_rla(
    module: &base::Module<IType>,
) -> Result<base::Module<theory::rla::RLA>, ()> {
    downcast_module(
        module,
        |dtype| theory::rla::Type::try_from(dtype),
        |itype| theory::rla::RLA::try_from(itype),
    )
}
