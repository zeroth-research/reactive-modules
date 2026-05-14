use crate::pytensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fmt;
use theory::Theory;

// ============================================================================
// DType enum (wire data types)
// ============================================================================

#[pyclass(frozen, eq, str)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DType {
    // at this moment, we keep the DType flat and encode the type
    // of elements in the names
    Bool(Vec<usize>),
    Int(Vec<usize>),
    Float(Vec<usize>),
    Real(Vec<usize>),
    UWord(u32),
    SWord(u32),
}

#[pymethods]
impl DType {
    /// Get the data dimensions of this data type
    #[getter]
    fn shape(&self) -> Vec<usize> {
        match &self {
            DType::Float(shape) => shape.clone(),
            DType::Int(shape) => shape.clone(),
            DType::Bool(shape) => shape.clone(),
            DType::Real(shape) => shape.clone(),
            DType::UWord(_) | DType::SWord(_) => vec![1],
        }
    }

    /// Create the same (Tensor) dtype but with a different shape
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        match self {
            DType::Bool(_) => Ok(DType::Bool(shape)),
            DType::Int(_) => Ok(DType::Int(shape)),
            DType::Float(_) => Ok(DType::Float(shape)),
            DType::Real(_) => Ok(DType::Real(shape)),
            DType::UWord(_) | DType::SWord(_) => {
                Err(PyValueError::new_err("cannot reshape word-level types"))
            }
        }
    }
}

fn fmt_comma_separated(f: &mut fmt::Formatter<'_>, items: &Vec<usize>) -> fmt::Result {
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{item}")?;
    }
    Ok(())
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Float(shape) => {
                write!(f, "Float(")?;
                fmt_comma_separated(f, shape)?;
                write!(f, ")")?;
            }
            DType::Int(shape) => {
                write!(f, "Int(")?;
                fmt_comma_separated(f, shape)?;
                write!(f, ")")?;
            }
            DType::Bool(shape) => {
                write!(f, "Bool(")?;
                fmt_comma_separated(f, shape)?;
                write!(f, ")")?;
            }
            DType::Real(shape) => {
                write!(f, "Real(")?;
                fmt_comma_separated(f, shape)?;
                write!(f, ")")?;
            }
            DType::UWord(n) => {
                write!(f, "UWord<{}>", n)?;
            }
            DType::SWord(n) => {
                write!(f, "SWord<{}>", n)?;
            }
        };
        Ok(())
    }
}

// ============================================================================
// IType enum (flat structure for PyO3 compatibility)
// ============================================================================

#[pyclass(str, frozen)]
#[derive(Debug, Clone)]
pub enum IType {
    // Arithmetic operations
    Add(),
    Sub(),
    Mul(),
    Div(),
    Mod(),
    Neg(),
    Abs(),
    MatMul(), // consider differentiation between matmul operators, or parameterisation.
    // To be designed with compliance with lower level platform in mind

    // Comparison operations
    Eq(),
    Neq(),
    Lt(),
    Le(),
    Gt(),
    Ge(),

    // Logical operations
    And(),
    Or(),
    Not(),
    Xor(),
    Xnor(),
    Implies(),

    // Control flow
    Ite(),

    // Transcendental functions
    Sin(),
    Cos(),

    // Special operations
    Id(),
    // index of maximal value in the flattened tensor
    Argmax(),
    // ReLU activation: max(0, x)
    ReLU(),
    // Tanh activation
    Tanh(),
    // Linear layer: output = input @ weight + bias
    // Reads: [input, weight, bias], Writes: [output]
    Linear(),

    // Tensor operations
    TensorGet(),
    TensorSet(),
    TensorSum(),
    TensorMean(),
    TensorMax(),
    // Concatenate N scalar/tensor inputs into a single tensor
    Stack(),

    // Word-level operations
    BitSelect(u32, u32),
    Extend(u32),
    ToBool(),
    ToWord1(),
    ToUnsigned(),
    ToSigned(),

    // Constants
    Tensor(PyTensor),
    ConstBool(bool),
    ConstInt(i64),

    // Symbol referring to uninterpreted constants or functions,
    // whose signature is known in the context, i.e., the current theory
    Uninterpreted(String),
}

impl fmt::Display for IType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IType::Add() => write!(f, "Add"),
            IType::Sub() => write!(f, "Sub"),
            IType::Mul() => write!(f, "Mul"),
            IType::Div() => write!(f, "Div"),
            IType::Mod() => write!(f, "Mod"),
            IType::Neg() => write!(f, "Neg"),
            IType::Abs() => write!(f, "Abs"),
            IType::MatMul() => write!(f, "MatMul"),
            IType::Sin() => write!(f, "Sin"),
            IType::Cos() => write!(f, "Cos"),
            IType::Eq() => write!(f, "Eq"),
            IType::Neq() => write!(f, "Neq"),
            IType::Lt() => write!(f, "Lt"),
            IType::Le() => write!(f, "Le"),
            IType::Gt() => write!(f, "Gt"),
            IType::Ge() => write!(f, "Ge"),
            IType::And() => write!(f, "And"),
            IType::Or() => write!(f, "Or"),
            IType::Not() => write!(f, "Not"),
            IType::Xor() => write!(f, "Xor"),
            IType::Xnor() => write!(f, "Xnor"),
            IType::Implies() => write!(f, "Implies"),
            IType::Ite() => write!(f, "Ite"),
            IType::Id() => write!(f, "Id"),
            IType::Argmax() => write!(f, "Argmax"),
            IType::ReLU() => write!(f, "ReLU"),
            IType::Tanh() => write!(f, "Tanh"),
            IType::Linear() => write!(f, "Linear"),
            IType::TensorGet() => write!(f, "TensorGet"),
            IType::TensorSet() => write!(f, "TensorSet"),
            IType::TensorSum() => write!(f, "TensorSum"),
            IType::TensorMean() => write!(f, "TensorMean"),
            IType::TensorMax() => write!(f, "TensorMax"),
            IType::Stack() => write!(f, "Stack"),
            IType::BitSelect(h, l) => write!(f, "BitSelect[{}:{}]", h, l),
            IType::Extend(n) => write!(f, "Extend({})", n),
            IType::ToBool() => write!(f, "ToBool"),
            IType::ToWord1() => write!(f, "ToWord1"),
            IType::ToUnsigned() => write!(f, "ToUnsigned"),
            IType::ToSigned() => write!(f, "ToSigned"),
            IType::Tensor(t) => {
                let flat = t.tensor.view([-1]);

                if let Ok(vals) = Vec::<f64>::try_from(&flat) {
                    let _ = write!(f, "Tensor([");
                    for (n, v) in vals.iter().take(5).enumerate() {
                        if n == 0 {
                            let _ = write!(f, "{}", v);
                        } else {
                            let _ = write!(f, " {}", v);
                        }
                    }
                    if flat.numel() > 3 {
                        let _ = write!(f, " ...");
                    }
                    write!(f, "])")
                } else {
                    write!(f, "Tensor({})", flat)
                }
            }
            IType::ConstBool(v) => write!(f, "Const: {}", v),
            IType::ConstInt(v) => write!(f, "Const: {}", v),
            IType::Uninterpreted(t) => write!(f, "{t}"),
        }
    }
}

impl Theory for IType {
    type DType = DType;

    fn type_check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a Self::DType>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
        Self::DType: 'a,
    {
        // FIXME
        Ok(())
    }
}

impl TryFrom<&DType> for theory::lia::Type {
    type Error = ();

    fn try_from(d: &DType) -> Result<Self, ()> {
        match d {
            DType::Int(shape) if shape.len() == 2 => Ok(theory::lia::Type::Int(shape[0], shape[1])),
            DType::Bool(shape) if shape.len() == 2 => Ok(theory::lia::Type::Bool(shape[0], shape[1])),
            _ => Err(()),
        }
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

impl TryFrom<&DType> for theory::rla::Type {
    type Error = ();

    fn try_from(d: &DType) -> Result<Self, ()> {
        match d {
            DType::Int(shape) if shape.len() == 2 => Ok(theory::rla::Type::Int(shape[0], shape[1])),
            DType::Bool(shape) if shape.len() == 2 => {
                Ok(theory::rla::Type::Bool(shape[0], shape[1]))
            }
            _ => Err(()),
        }
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
            IType::ConstInt(v) => RLA::ConstInt(vec![vec![*v]]),
            IType::ConstBool(b) => RLA::ConstBool(vec![vec![*b]]),
            _ => return Err(()),
        })
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
