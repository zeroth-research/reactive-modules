use crate::pyval::PyVal;
use crate::torch::wrappedcontext::WrappedContext;
use crate::torch::wrappedterm::WrappedTerm;
use torch::{DType, IType, TorchTerm};

use pyo3::prelude::*;
use pyo3::types::PyList;

use base::Atom;
use base::wire::Interface;

use crate::util::str_to_pyerr;

type Intf = Interface<DType>;

#[pyclass]
pub struct WrappedAtom {
    pub(crate) atom: Atom<DType, IType>,
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
    result: &mut Vec<TorchTerm>,
) -> Vec<usize> {
    let mut args: Vec<usize> = vec![];

    // FIXME: get rid of the code duplication
    for r in pyvals {
        match r {
            PyVal::Sym(var_id, _) => args.push(*var_id),
            PyVal::Tensor(tensor) => {
                let var = ctx.fresh_var();
                result.push(TorchTerm::new_unchecked(
                    IType::Const(tensor.tensor.shallow_clone()),
                    Interface::single(var, DType::Tensor),
                    Interface::empty(),
                ));
                args.push(var);
            }
            //PyVal::F64(val) => {
            //    let var = ctx.fresh_var();
            //    result.push(TorchTerm::new(
            //        IType::Const(tch::Tensor::from_slice(&[*val])),
            //        Wire::one(var, DType::Tensor),
            //        Wire::none(),
            //    ));
            //    args.push(var);
            //}
            PyVal::Int(val) => {
                let var = ctx.fresh_var();
                result.push(TorchTerm::new_unchecked(
                    IType::Const(tch::Tensor::from_slice(&[*val])),
                    Interface::single(var, DType::Tensor),
                    Interface::empty(),
                ));
                args.push(var);
            }
            PyVal::Bool(val) => {
                let var = ctx.fresh_var();
                result.push(TorchTerm::new_unchecked(
                    IType::Const(tch::Tensor::from_slice(&[*val])),
                    Interface::single(var, DType::Tensor),
                    Interface::empty(),
                ));
                args.push(var);
            }
        }
    }

    args
}

/// Translate a list of [WrappedTerm]s into a vector of [TorchTerm]s.
/// The results is wrapped in [PyResult] so that we can easily propagate errors
/// back to Python.
///
/// The resulting vector can be longer that the input vector, because we may generate new
/// [TorchTerm] terms that represent read or written constants (these we can represent as [PyVal]s
/// in `WrappedTerm.reads` and `WrappedTerm.writes`, but in [TorchTerm]s the wires cannot take "constant"
/// values.)
pub fn wterms_to_torchterms(
    ctx: &mut WrappedContext,
    terms: &Bound<'_, PyList>,
) -> PyResult<Vec<TorchTerm>> {
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

        let write = Interface::sequence(wargs.into_iter().map(|val| (val, DType::Tensor)))
            .map_err(str_to_pyerr)?;
        let read = Interface::sequence(rargs.into_iter().map(|val| (val, DType::Tensor)))
            .map_err(str_to_pyerr)?;
        result.push(TorchTerm::new_unchecked(wterm.op.clone(), write, read));
    }

    Ok(result)
}

/// Translate a list of `PyVal::Sym` values into a wiring. This is a simple version of
/// [process_pyvals] where we assume no constants are in the input vector.
pub(crate) fn vars_to_wiring(vals: &Bound<'_, PyList>) -> PyResult<Intf> {
    let mut names: Vec<usize> = Vec::new();
    for item in vals {
        let pyval: &Bound<'_, PyVal> = item.downcast()?;
        let pyval = pyval.borrow();
        match &*pyval {
            PyVal::Sym(var_id, _) => names.push(*var_id),
            _ => panic!("Invalid input variable: {:?}", pyval),
        }
    }

    Ok(
        Interface::sequence(names.into_iter().map(|val| (val, DType::Tensor)))
            .map_err(str_to_pyerr)?,
    )
}

#[pymethods]
impl WrappedAtom {
    //#[new]
    //fn new(
    //    ctx: &Bound<'_, WrappedContext>,
    //    // variables
    //    latched: &Bound<'_, PyList>,
    //    next: &Bound<'_, PyList>,
    //    // init and update
    //    init: &Bound<'_, PyList>,
    //    update: &Bound<'_, PyList>,
    //) -> Self {
    //    let latched = vars_to_wiring(latched).unwrap();
    //    let next = vars_to_wiring(next).unwrap();
    //
    //    let ctx: &mut WrappedContext = &mut ctx.borrow_mut();
    //    let init = wterms_to_torchterms(ctx, init).unwrap();
    //    let update = wterms_to_torchterms(ctx, update).unwrap();
    //
    //    let atom = Atom::sequential(&[latched, next], init, update).unwrap();
    //    Self { atom }
    //}

    fn dbg(&self) {
        println!("{}", self.atom);
    }
}
