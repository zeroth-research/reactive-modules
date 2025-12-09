mod wrappedcontext;
mod wrappedmodule;
mod wrappedterm;

pub use wrappedcontext::WrappedContext;
pub use wrappedmodule::WrappedModule;
pub use wrappedterm::WrappedTerm;

use crate::pyval::PyVal;
use crate::util::str_to_pyerr;

use base::wire::Interface;
use toy::val::Val;
use toy::{DType, IType, ToyTerm};
type Intf = Interface<DType>;

use pyo3::prelude::*;
use pyo3::types::PyList;

fn new_term(op: IType, reads: Intf, writes: Intf) -> Result<ToyTerm, &'static str> {
    Ok(ToyTerm::new_unchecked(op, writes, reads))
}

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
    result: &mut Vec<ToyTerm>,
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
                    Interface::empty(),
                    Interface::single(var, DType::Real),
                )
                .expect("Failed creating a constant term for a pyval");
                result.push(term);
                args.push((var, DType::Real));
            }
            PyVal::Int(val) => {
                let var = ctx.ctx.tmp_var(DType::Int);
                let term = new_term(
                    IType::Const(Val::Int(*val)),
                    Interface::empty(),
                    Interface::single(var, DType::Int),
                )
                .expect("Failed creating a constant term for a pyval");
                result.push(term);
                args.push((var, DType::Int));
            }
            PyVal::Bool(val) => {
                let var = ctx.ctx.tmp_var(DType::Bool);
                let term = new_term(
                    IType::Const(Val::Bool(*val)),
                    Interface::empty(),
                    Interface::single(var, DType::Bool),
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
fn wterms_to_terms(ctx: &mut WrappedContext, terms: &Bound<'_, PyList>) -> PyResult<Vec<ToyTerm>> {
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

        let write = Interface::sequence(wargs.into_iter()).map_err(str_to_pyerr)?;
        let read = Interface::sequence(rargs.into_iter()).map_err(str_to_pyerr)?;
        let term = new_term(wterm.op.clone(), read, write).map_err(str_to_pyerr)?;
        result.push(term);
    }

    Ok(result)
}

/// Translate a list of `PyVal::Sym` values into a wiring. This is a simple version of
/// [process_pyvals] where we assume no constants are in the input vector.
pub(crate) fn vars_to_wiring(vals: &Bound<'_, PyList>) -> PyResult<Intf> {
    let mut names: Vec<(usize, String)> = Vec::new();
    for item in vals {
        let pyval: &Bound<'_, PyVal> = item.downcast()?;
        let pyval = pyval.borrow();
        match &*pyval {
            PyVal::Sym(var_id, ty) => names.push((*var_id, ty.clone())),
            _ => panic!("Invalid input variable: {:?}", pyval),
        }
    }

    Ok(Interface::sequence(
        names
            .into_iter()
            .map(|(val, ty)| (val, ty.parse().expect("Invalid type"))),
    )
    .map_err(str_to_pyerr)?)
}
