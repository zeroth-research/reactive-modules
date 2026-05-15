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

    fn is_bool(&self) -> bool {
        matches!(self, DType::Bool(_))
    }
    fn is_int(&self) -> bool {
        matches!(self, DType::Int(_))
    }
    fn is_float(&self) -> bool {
        matches!(self, DType::Float(_))
    }
    fn is_real(&self) -> bool {
        matches!(self, DType::Real(_))
    }
    fn is_bv(&self) -> bool {
        matches!(self, DType::UWord(_) | DType::SWord(_))
    }

    fn bv_bw(&self) -> PyResult<u32> {
        match self {
            DType::UWord(bw) | DType::SWord(bw) => Ok(*bw),
            _ => Err(pyo3::exceptions::PyTypeError::new_err("not a BV type")),
        }
    }

    fn bv_signed(&self) -> PyResult<bool> {
        match self {
            DType::SWord(_) => Ok(true),
            DType::UWord(_) => Ok(false),
            _ => Err(pyo3::exceptions::PyTypeError::new_err("not a BV type")),
        }
    }
}

fn shape2(shape: &[usize]) -> Result<(usize, usize), ()> {
    match shape {
        [n] => Ok((1, *n)),
        [m, n] => Ok((*m, *n)),
        _ => Err(()),
    }
}

fn is_scalar_shape(s: &[usize]) -> bool {
    matches!(shape2(s), Ok((1, 1)))
}

impl DType {
    pub fn is_scalar(&self) -> bool {
        matches!(shape2(self.shape().as_slice()), Ok((1, 1)))
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

// ============================================================================
// Theory impl
// ============================================================================

fn tc_next<'a, I, D>(iter: &mut I, op: &str, i: usize) -> Result<&'a DType, String>
where
    I: Iterator<Item = D>,
    D: TryInto<&'a DType>,
{
    if let Some(d) = iter.next() {
        d.try_into()
            .map_err(|_| format!("{op}: arg {i} not compatible with DType"))
    } else {
        Err(format!("{op}: arg {i} expected but got none"))
    }
}

fn same_base_kind(a: &DType, b: &DType) -> bool {
    matches!(
        (a, b),
        (DType::Bool(_), DType::Bool(_))
            | (DType::Int(_), DType::Int(_))
            | (DType::Float(_), DType::Float(_))
            | (DType::Real(_), DType::Real(_))
            | (DType::UWord(_), DType::UWord(_))
            | (DType::SWord(_), DType::SWord(_))
    )
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
        use DType::*;
        let mut rd = read.into_iter();
        let mut wr = write.into_iter();
        let op = self.to_string();
        let op = op.as_str();

        macro_rules! rd {
            ($i:expr) => {
                tc_next(&mut rd, op, $i)?
            };
        }
        macro_rules! wr {
            ($i:expr) => {
                tc_next(&mut wr, op, $i)?
            };
        }
        macro_rules! no_more_args {
            () => {
                if rd.next().is_some() {
                    return Err(format!("{op}: too many read args"));
                }
                if wr.next().is_some() {
                    return Err(format!("{op}: too many write args"));
                }
            };
        }

        match self {
            // -- identity / control flow ------------------------------------------
            IType::Id() => {
                let r = rd!(0);
                let w = wr!(0);
                no_more_args!();
                if r != w {
                    return Err(format!("{op}: read type {r} != write type {w}"));
                }
                Ok(())
            }
            IType::Ite() => {
                let (r0, r1, r2) = (rd!(0), rd!(1), rd!(2));
                let w0 = wr!(0);
                no_more_args!();
                let Bool(cond_shape) = r0 else {
                    return Err(format!("{op}: condition must be Bool, got {r0}"));
                };
                if !is_scalar_shape(cond_shape) {
                    return Err(format!(
                        "{op}: condition must be Bool scalar, got Bool({cond_shape:?})"
                    ));
                }
                if r1 != r2 {
                    return Err(format!(
                        "{op}: branches must have same type, got {r1} and {r2}"
                    ));
                }
                if w0 != r1 {
                    return Err(format!("{op}: output {w0} must match branch type {r1}"));
                }
                Ok(())
            }

            // -- comparison -------------------------------------------------------
            IType::Eq() | IType::Neq() | IType::Lt() | IType::Le() | IType::Gt() | IType::Ge() => {
                let (r0, r1) = (rd!(0), rd!(1));
                let w0 = wr!(0);
                no_more_args!();
                if r0 != r1 {
                    return Err(format!(
                        "{op}: inputs must have the same type, got {r0} and {r1}"
                    ));
                }
                if matches!(r0, Bool(_)) {
                    return Err(format!("{op}: inputs cannot be Bool"));
                }
                let Bool(w_shape) = w0 else {
                    return Err(format!("{op}: output must be Bool, got {w0}"));
                };
                let r_shape = r0.shape();
                if w_shape.as_slice() != r_shape.as_slice() {
                    return Err(format!(
                        "{op}: output Bool shape {w_shape:?} != input shape {r_shape:?}"
                    ));
                }
                Ok(())
            }

            // -- logical ----------------------------------------------------------
            IType::And() | IType::Or() | IType::Xor() | IType::Xnor() | IType::Implies() => {
                let (r0, r1) = (rd!(0), rd!(1));
                let w0 = wr!(0);
                no_more_args!();
                let (Bool(_), Bool(_), Bool(_)) = (r0, r1, w0) else {
                    return Err(format!("{op}: all operands must be Bool"));
                };
                if r0 != r1 {
                    return Err(format!(
                        "{op}: inputs must have the same shape, got {r0} and {r1}"
                    ));
                }
                if w0 != r0 {
                    return Err(format!("{op}: output {w0} must match input type {r0}"));
                }
                Ok(())
            }
            IType::Not() => {
                let r = rd!(0);
                let w = wr!(0);
                no_more_args!();
                let (Bool(_), Bool(_)) = (r, w) else {
                    return Err(format!("{op}: operands must be Bool"));
                };
                if r != w {
                    return Err(format!(
                        "{op}: input {r} and output {w} must have the same shape"
                    ));
                }
                Ok(())
            }

            // -- arithmetic (binary) ---------------------------------------------
            IType::Add() | IType::Sub() | IType::Mul() | IType::Div() | IType::Mod() => {
                let (r0, r1) = (rd!(0), rd!(1));
                let w0 = wr!(0);
                no_more_args!();
                if matches!(r0, Bool(_)) {
                    return Err(format!(
                        "{op}: inputs must be numeric (Int/Float/Real/UWord/SWord)"
                    ));
                }
                if r0 != r1 {
                    return Err(format!(
                        "{op}: inputs must have the same type, got {r0} and {r1}"
                    ));
                }
                if w0 != r0 {
                    return Err(format!("{op}: output {w0} must match input type {r0}"));
                }
                Ok(())
            }

            // -- arithmetic (unary) ----------------------------------------------
            IType::Neg() | IType::Abs() => {
                let r = rd!(0);
                let w = wr!(0);
                no_more_args!();
                if matches!(r, Bool(_)) {
                    return Err(format!(
                        "{op}: input must be numeric (Int/Float/Real/UWord/SWord)"
                    ));
                }
                if r != w {
                    return Err(format!(
                        "{op}: input {r} and output {w} must have the same type"
                    ));
                }
                Ok(())
            }

            // -- matrix multiply -------------------------------------------------
            IType::MatMul() => {
                let (r0, r1) = (rd!(0), rd!(1));
                let w0 = wr!(0);
                no_more_args!();
                if !same_base_kind(r0, r1) || !same_base_kind(r0, w0) {
                    return Err(format!("{op}: all operands must have the same base type"));
                }
                if matches!(r0, Bool(_) | UWord(_) | SWord(_)) {
                    return Err(format!("{op}: operands must be numeric (Int/Float/Real)"));
                }
                let r0_shape = r0.shape();
                let r1_shape = r1.shape();
                let w0_shape = w0.shape();
                let (m, k) = shape2(&r0_shape).map_err(|_| format!("{op}: lhs must be 1-D or 2-D"))?;
                // 1-D rhs treated as column vector (k, 1), matching PyTorch A @ v semantics
                let (k2, n) = if r1_shape.len() == 1 {
                    (r1_shape[0], 1)
                } else {
                    shape2(&r1_shape).map_err(|_| format!("{op}: rhs must be 1-D or 2-D"))?
                };
                // 1-D output treated as column (m, 1), result squeezed to [m]
                let (om, on) = if w0_shape.len() == 1 {
                    (w0_shape[0], 1)
                } else {
                    shape2(&w0_shape).map_err(|_| format!("{op}: output must be 1-D or 2-D"))?
                };
                if k != k2 {
                    return Err(format!("{op}: inner dimensions mismatch: {k} != {k2}"));
                }
                if m != om || n != on {
                    return Err(format!("{op}: output shape ({om},{on}) must be ({m},{n})"));
                }
                Ok(())
            }

            // -- transcendental --------------------------------------------------
            IType::Sin() | IType::Cos() => {
                let r = rd!(0);
                let w = wr!(0);
                no_more_args!();
                if !matches!(r, Real(_)) {
                    return Err(format!("{op}: input must be Real, got {r}"));
                }
                if r != w {
                    return Err(format!(
                        "{op}: input {r} and output {w} must have the same type"
                    ));
                }
                Ok(())
            }

            // -- NN --------------------------------------------------------------
            IType::ReLU() | IType::Tanh() => {
                let r = rd!(0);
                let w = wr!(0);
                no_more_args!();
                if !matches!(r, Float(_) | Real(_)) {
                    return Err(format!("{op}: input must be Float or Real, got {r}"));
                }
                if r != w {
                    return Err(format!(
                        "{op}: input {r} and output {w} must have the same type"
                    ));
                }
                Ok(())
            }
            IType::Linear() => {
                let (r0, r1, r2) = (rd!(0), rd!(1), rd!(2)); // input, weight, bias
                let w0 = wr!(0);
                no_more_args!();
                let (Float(inp_s), Float(wgt_s), Float(bias_s), Float(out_s)) = (r0, r1, r2, w0)
                else {
                    return Err(format!("{op}: all operands must be Float"));
                };
                let (batch, in_f) = shape2(inp_s).map_err(|_| format!("{op}: input must be 1-D or 2-D"))?;
                let (out_f, wgt_in) = shape2(wgt_s).map_err(|_| format!("{op}: weight must be 1-D or 2-D"))?;
                let (_, bias_cols) = shape2(bias_s).map_err(|_| format!("{op}: bias must be 1-D or 2-D"))?;
                let (out_batch, out_cols) = shape2(out_s).map_err(|_| format!("{op}: output must be 1-D or 2-D"))?;
                if in_f != wgt_in {
                    return Err(format!(
                        "{op}: input features {in_f} != weight cols {wgt_in}"
                    ));
                }
                if out_f != bias_cols {
                    return Err(format!(
                        "{op}: weight rows {out_f} != bias cols {bias_cols}"
                    ));
                }
                if batch != out_batch || out_f != out_cols {
                    return Err(format!(
                        "{op}: output must be Float({batch}, {out_f}), got Float({out_batch}, {out_cols})"
                    ));
                }
                Ok(())
            }

            // -- tensor ops ------------------------------------------------------
            IType::Argmax() => {
                let _r = rd!(0);
                let w0 = wr!(0);
                no_more_args!();
                if !matches!(w0, Int(s) if is_scalar_shape(s)) {
                    return Err(format!("{op}: output must be Int scalar, got {w0}"));
                }
                Ok(())
            }
            IType::TensorSum() | IType::TensorMean() | IType::TensorMax() => {
                let r0 = rd!(0);
                let w0 = wr!(0);
                no_more_args!();
                let ok = match (r0, w0) {
                    (Float(_), Float(s)) | (Int(_), Int(s)) | (Real(_), Real(s)) => is_scalar_shape(s),
                    _ => false,
                };
                if !ok {
                    return Err(format!(
                        "{op}: output must be scalar of same base type as input, got {w0}"
                    ));
                }
                Ok(())
            }
            IType::TensorGet() => {
                let (r0, r1) = (rd!(0), rd!(1)); // tensor, index
                let w0 = wr!(0);
                no_more_args!();
                if !matches!(r1, Int(s) if is_scalar_shape(s)) {
                    return Err(format!("{op}: index must be Int scalar, got {r1}"));
                }
                let ok = match (r0, w0) {
                    (Float(_), Float(s)) | (Int(_), Int(s)) | (Real(_), Real(s)) => is_scalar_shape(s),
                    _ => false,
                };
                if !ok {
                    return Err(format!(
                        "{op}: output must be scalar of same base type as tensor input, got {w0}"
                    ));
                }
                Ok(())
            }
            IType::TensorSet() => {
                let (r0, r1, r2) = (rd!(0), rd!(1), rd!(2)); // tensor, index, value
                let w0 = wr!(0);
                no_more_args!();
                if !matches!(r1, Int(s) if is_scalar_shape(s)) {
                    return Err(format!("{op}: index must be Int scalar, got {r1}"));
                }
                let r2_ok = match (r0, r2) {
                    (Float(_), Float(s)) | (Int(_), Int(s)) | (Real(_), Real(s)) => is_scalar_shape(s),
                    _ => false,
                };
                if !r2_ok {
                    return Err(format!(
                        "{op}: value must be scalar of same base type as tensor, got {r2}"
                    ));
                }
                if r0 != w0 {
                    return Err(format!(
                        "{op}: output {w0} must match input tensor type {r0}"
                    ));
                }
                Ok(())
            }

            // -- BV / casting ----------------------------------------------------
            IType::BitSelect(h, l) => {
                let r0 = rd!(0);
                let w0 = wr!(0);
                no_more_args!();
                if !matches!(r0, UWord(_) | SWord(_)) {
                    return Err(format!("{op}: input must be UWord or SWord, got {r0}"));
                }
                let out_bw = h - l + 1;
                if *w0 != UWord(out_bw) {
                    return Err(format!("{op}: output must be UWord({out_bw}), got {w0}"));
                }
                Ok(())
            }
            IType::Extend(w) => {
                let r0 = rd!(0);
                let w0 = wr!(0);
                no_more_args!();
                let expected = match r0 {
                    UWord(bw) => {
                        if bw > w {
                            return Err(format!("{op}: input width {bw} > target {w}"));
                        }
                        UWord(*w)
                    }
                    SWord(bw) => {
                        if bw > w {
                            return Err(format!("{op}: input width {bw} > target {w}"));
                        }
                        SWord(*w)
                    }
                    _ => return Err(format!("{op}: input must be UWord or SWord, got {r0}")),
                };
                if *w0 != expected {
                    return Err(format!("{op}: output must be {expected}, got {w0}"));
                }
                Ok(())
            }
            IType::ToBool() => {
                let r0 = rd!(0);
                let w0 = wr!(0);
                no_more_args!();
                if !matches!(r0, UWord(_) | SWord(_)) {
                    return Err(format!("{op}: input must be UWord or SWord, got {r0}"));
                }
                if !matches!(w0, Bool(s) if is_scalar_shape(s)) {
                    return Err(format!("{op}: output must be Bool scalar, got {w0}"));
                }
                Ok(())
            }
            IType::ToWord1() => {
                let r0 = rd!(0);
                let w0 = wr!(0);
                no_more_args!();
                if !matches!(r0, Bool(_) | UWord(_) | SWord(_)) {
                    return Err(format!(
                        "{op}: input must be Bool, UWord, or SWord, got {r0}"
                    ));
                }
                if *w0 != UWord(1) {
                    return Err(format!("{op}: output must be UWord(1), got {w0}"));
                }
                Ok(())
            }
            IType::ToUnsigned() => {
                let r0 = rd!(0);
                let w0 = wr!(0);
                no_more_args!();
                let SWord(bw) = r0 else {
                    return Err(format!("{op}: input must be SWord, got {r0}"));
                };
                if *w0 != UWord(*bw) {
                    return Err(format!("{op}: output must be UWord({bw}), got {w0}"));
                }
                Ok(())
            }
            IType::ToSigned() => {
                let r0 = rd!(0);
                let w0 = wr!(0);
                no_more_args!();
                let UWord(bw) = r0 else {
                    return Err(format!("{op}: input must be UWord, got {r0}"));
                };
                if *w0 != SWord(*bw) {
                    return Err(format!("{op}: output must be SWord({bw}), got {w0}"));
                }
                Ok(())
            }

            // -- constants -------------------------------------------------------
            IType::Tensor(t) => {
                let w0 = wr!(0);
                no_more_args!();
                let sz = t.tensor.size();
                let expected_shape: Vec<usize> = sz.iter().map(|&x| x as usize).collect();
                match w0 {
                    Float(s) | Int(s) | Real(s) | Bool(s) => {
                        if *s != expected_shape {
                            return Err(format!(
                                "{op}: write shape {s:?} doesn't match tensor shape {expected_shape:?}"
                            ));
                        }
                    }
                    _ => return Err(format!("{op}: write type must be Float, Int, or Real")),
                }
                Ok(())
            }
            IType::ConstBool(_) => {
                let w0 = wr!(0);
                no_more_args!();
                if !matches!(w0, Bool(s) if is_scalar_shape(s)) {
                    return Err(format!("{op}: output must be Bool scalar, got {w0}"));
                }
                Ok(())
            }
            IType::ConstInt(_) => {
                let w0 = wr!(0);
                no_more_args!();
                if !matches!(w0, Int(s) | Real(s) if is_scalar_shape(s))
                    && !matches!(w0, UWord(_) | SWord(_))
                {
                    return Err(format!(
                        "{op}: output must be Int/Real scalar, UWord, or SWord, got {w0}"
                    ));
                }
                Ok(())
            }

            // -- uninterpreted ---------------------------------------------------
            IType::Uninterpreted(_) => Ok(()),
        }
    }
}

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
            DType::Real(shape) if shape.len() == 2 => Ok(theory::rla::Type::Real(shape[0], shape[1])),
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
            IType::ConstInt(v) => RLA::ConstReal(vec![vec![*v as f64]]),
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
