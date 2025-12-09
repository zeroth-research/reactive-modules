use std::fmt;

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::PyVal;
use crate::smt::wrappedcontext::WrappedContext;
use crate::smt::wrappedterm::WrappedTerm;
use crate::util::str_to_pyerr;

use base::wire::Wire;

use smt::dtype::DType;
use smt::itype::{IType, Val};

type WireTy = Wire<DType>;
type SmtAtom = base::Atom<DType, IType>;
type SmtTerm = base::Term<DType, IType>;

#[pyclass]
pub struct WrappedAtom {
    pub(crate) atom: SmtAtom,
}

fn new_term(op: IType, reads: Wire<DType>, writes: Wire<DType>) -> Result<SmtTerm, &'static str> {
    Ok(SmtTerm::new(op, writes, reads))
}

// It is safe to share this struct for the same reasons as for PyTensor
unsafe impl Sync for WrappedAtom {}

// TODO: the following functions that transfer Python data to our structures may be optimized

/// Translate a vector of [PyVal]s into a vector of wire identifiers (`Vec<usize>`).
///
/// This function is used when translating [WrappedTerm]s into [Term](base::term::Term)s. This translation
/// is using a vector `result` where the translated [Term](base::term::Term)s are stored.
/// This vector is passed also to this function and used as described below.
///
/// The values from the input vector `pyvals` get translated as follows:
/// `PyVal::Sym(x)` gets translated into the identifier `x`, and other [PyVal]
/// values (constants) get translated into a new nullary term `IType::Const` that is added
/// to `result` and the output wire of this term is used as the translated identifier.
/// This new term simply returns the value of this constant. This is to workaround the fact that
/// we do not have wires that represent constants.
///
/// # Examples
///
///  If the input is `[PyVal::Sym(2), PyVal::I64(7)]`, then when translating
/// `PyVal::I64(7)`, a term `Term { IType::Const, reads: [], writes: [19]}` gets generated
/// and added to `results`, where 19 is just an example value. This value is obtained
/// from the `ctx` object. The translated vector returned from this function is then `[2, 19]`.
fn process_pyvals(
    ctx: &mut WrappedContext,
    pyvals: &Vec<PyVal>,
    result: &mut Vec<SmtTerm>,
) -> Vec<(usize, DType)> {
    let mut args: Vec<(usize, DType)> = vec![];

    // FIXME: get rid of the code duplication
    for r in pyvals {
        match r {
            PyVal::Sym(var_id, ty) => {
                let ty: DType = ty.parse().expect("Failed parsing DType");
                args.push((*var_id, ty));
            }
            PyVal::Real(val) => {
                let var = ctx.ctx.tmp_var(DType::Real);
                let term = new_term(
                    IType::Const(Val::Real(*val)),
                    Wire::none(),
                    Wire::one(var, DType::Real),
                )
                .expect("Failed creating a constant term for a pyval");
                result.push(term);
                args.push((var, DType::Real));
            }
            PyVal::Int(val) => {
                let var = ctx.ctx.tmp_var(DType::Int);
                let term = new_term(
                    IType::Const(Val::Int(*val)),
                    Wire::none(),
                    Wire::one(var, DType::Int),
                )
                .expect("Failed creating a constant term for a pyval");
                result.push(term);
                args.push((var, DType::Int));
            }
            PyVal::Bool(val) => {
                let var = ctx.ctx.tmp_var(DType::Bool);
                let term = new_term(
                    IType::Const(Val::Bool(*val)),
                    Wire::none(),
                    Wire::one(var, DType::Bool),
                )
                .expect("Failed creating a term for a boolean value");
                result.push(term);
                args.push((var, DType::Bool));
            }
            #[cfg(feature = "enable-torch")]
            PyVal::Tensor(_) => {
                // translate to a matrix if the tensor is a matrix. Otherwise fail
                todo!()
            }
        }
    }

    args
}

/// Translate a list of [WrappedTerm]s into a vector of [Term]s.
/// The results is wrapped in [PyResult] so that we can easily propagate errors
/// back to Python.
///
/// The resulting vector can be longer that the input vector, because we may generate new
/// [Term] terms that represent read or written constants (these we can represent as [PyVal]s
/// in `WrappedTerm.reads` and `WrappedTerm.writes`, but in [Term]s the wires cannot take "constant"
/// values.)
fn wterms_to_terms(ctx: &mut WrappedContext, terms: &Bound<'_, PyList>) -> PyResult<Vec<SmtTerm>> {
    let mut result = vec![];

    for item in terms {
        let wterm: &Bound<'_, WrappedTerm> = item.downcast()?;
        let wterm = wterm.borrow();

        // Translate `wterm.reads` and `wterm.writes` into vectors of wire identifiers.
        // Because constants have no wire identifiers, we translate each constant
        // into a term and we use the identifier of this term's output wire in place of this
        // constant. See the docstring of [process_pyvals].
        let rargs = process_pyvals(ctx, &wterm.reads, &mut result);
        let wargs = process_pyvals(ctx, &wterm.writes, &mut result);

        let write = Wire::from_iter(wargs.into_iter());
        let read = Wire::from_iter(rargs.into_iter());
        let term = new_term(wterm.op.clone(), read, write).map_err(str_to_pyerr)?;
        result.push(term);
    }

    Ok(result)
}

/// Translate a list of `PyVal::Sym` values into a wiring. This is a simple version of
/// [process_pyvals] where we assume no constants are in the input vector.
pub(crate) fn vars_to_wiring(vals: &Bound<'_, PyList>) -> PyResult<WireTy> {
    let mut names: Vec<(usize, String)> = Vec::new();
    for item in vals {
        let pyval: &Bound<'_, PyVal> = item.downcast()?;
        let pyval = pyval.borrow();
        match &*pyval {
            PyVal::Sym(var_id, ty) => names.push((*var_id, ty.clone())),
            _ => panic!("Invalid input variable: {:?}", pyval),
        }
    }

    Ok(Wire::from_iter(
        names
            .into_iter()
            .map(|(val, ty)| (val, ty.parse().expect("Invalid type"))),
    ))
}

#[pymethods]
impl WrappedAtom {
    #[new]
    fn new(
        ctx: &Bound<'_, WrappedContext>,
        read: &Bound<'_, PyList>,
        write: &Bound<'_, PyList>,
        init: &Bound<'_, PyList>,
        update: &Bound<'_, PyList>,
    ) -> Self {
        let read = vars_to_wiring(read).unwrap();
        let write = vars_to_wiring(write).unwrap();

        let ctx: &mut WrappedContext = &mut ctx.borrow_mut();
        let init = wterms_to_terms(ctx, init).unwrap();
        let update = wterms_to_terms(ctx, update).unwrap();

        let atom = SmtAtom::sequential(read, write, init, update);
        Self { atom }
    }

    fn dbg(&self) {
        println!("{}", self.atom);
    }
}

impl std::fmt::Debug for WrappedAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Reads: {:?}", self.atom.read())?;
        writeln!(f, "Controls: {:?}", self.atom.ctrl())?;
        writeln!(f, "Awaits: NOT IMPLEMENTED {:?}", self.atom.wait())?;
        writeln!(f, "-------------")?;
        writeln!(f, "Init:")?;
        for term in self.atom.init() {
            writeln!(f, "  {:?}", term)?;
        }
        writeln!(f, "-------------")?;
        writeln!(f, "Update:")?;
        for term in self.atom.update() {
            writeln!(f, "  {:?}", term)?;
        }
        writeln!(f, "-------------")
    }
}
