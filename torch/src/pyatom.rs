use std::fmt;

use crate::context::Context;
use crate::pyterm::PyTerm;
use crate::pyval::PyVal;
use crate::term::{TorchDType, TorchOp, TorchTerm};

use pyo3::prelude::*;
use pyo3::types::PyList;

use base::Atom;
use base::Wire;

type WireTy = Wire<TorchDType>;

#[pyclass]
pub struct PyAtom {
    atom: Atom<TorchDType, TorchOp>,
}

// It is safe to share this struct for the same reasons as for PyTensor
unsafe impl Sync for PyAtom {}

// TODO: the following functions that transfer Python data to our structures may be optimized

/// Translate a vector of [PyVal]s into a vector of wire identifiers (`Vec<usize>`).
///
/// This function is used when translating [PyTerm]s into [Term](base::term::Term)s. This translation
/// is using a vector `result` where the translated [Term](base::term::Term)s are stored.
/// This vector is passed also to this function and used as described below.
///
/// The values from the input vector `pyvals` get translated as follows:
/// `PyVal::Sym(x)` gets translated into the identifier `x`, and other [PyVal]
/// values (constants) get translated into a new nullary term `TorchOp::Const` that is added
/// to `result` and the output wire of this term is used as the translated identifier.
/// This new term simply returns the value of this constant. This is to workaround the fact that
/// we do not have wires that represent constants.
///
/// # Examples
///
///  If the input is `[PyVal::Sym(2), PyVal::I64(7)]`, then when translating
/// `PyVal::I64(7)`, a term `Term { TorchOp::Const, reads: [], writes: [19]}` gets generated
/// and added to `results`, where 19 is just an example value. This value is obtained
/// from the `ctx` object. The translated vector returned from this function is then `[2, 19]`.
fn process_pyvals(
    ctx: &mut Context,
    pyvals: &Vec<PyVal>,
    result: &mut Vec<TorchTerm>,
) -> Vec<usize> {
    let mut args: Vec<usize> = vec![];

    // FIXME: get rid of the code duplication
    for r in pyvals {
        match r {
            PyVal::Sym(var_id) => args.push(*var_id),
            PyVal::Tensor(tensor) => {
                let var = ctx.fresh_var();
                result.push(TorchTerm::new(
                    TorchOp::Const(tensor.tensor.shallow_clone()),
                    Wire::one(var, "tensor"),
                    Wire::none(),
                ));
                args.push(var);
            }
            PyVal::F64(val) => {
                let var = ctx.fresh_var();
                result.push(TorchTerm::new(
                    TorchOp::Const(tch::Tensor::from_slice(&[*val])),
                    Wire::one(var, "tensor"),
                    Wire::none(),
                ));
                args.push(var);
            }
            PyVal::I64(val) => {
                let var = ctx.fresh_var();
                result.push(TorchTerm::new(
                    TorchOp::Const(tch::Tensor::from_slice(&[*val])),
                    Wire::one(var, "tensor"),
                    Wire::none(),
                ));
                args.push(var);
            }
        }
    }

    args
}

/// Translate a list of [PyTerm]s into a vector of [TorchTerm]s.
/// The results is wrapped in [PyResult] so that we can easily propagate errors
/// back to Python.
///
/// The resulting vector can be longer that the input vector, because we may generate new
/// [TorchTerm] terms that represent read or written constants (these we can represent as [PyVal]s
/// in `PyTerm.reads` and `PyTerm.writes`, but in [TorchTerm]s the wires cannot take "constant"
/// values.)
fn pyterms_to_torchterms(ctx: &mut Context, terms: &Bound<'_, PyList>) -> PyResult<Vec<TorchTerm>> {
    let mut result = vec![];

    for item in terms {
        let pyterm: &Bound<'_, PyTerm> = item.downcast()?;
        let pyterm = pyterm.borrow();

        // Translate `pyterm.reads` and `pyterm.writes` into vectors of wire identifiers.
        // Because constants have no wire identifiers, we translate each constant
        // into a term and we use the identifier of this term's output wire in place of this
        // constant. See the docstring of [process_pyvals].
        let rargs = process_pyvals(ctx, &pyterm.reads, &mut result);
        let wargs = process_pyvals(ctx, &pyterm.writes, &mut result);

        let write = Wire::from_iter(wargs.into_iter().map(|val| (val, "tensor")));
        let read = Wire::from_iter(rargs.into_iter().map(|val| (val, "tensor")));
        result.push(TorchTerm::new(pyterm.op.clone(), write, read));
    }

    Ok(result)
}

/// Translate a list of `PyVal::Sym` values into a wiring. This is a simple version of
/// [process_pyvals] where we assume no constants are in the input vector.
fn vars_to_wiring(vals: &Bound<'_, PyList>) -> PyResult<WireTy> {
    let mut names: Vec<usize> = Vec::new();
    for item in vals {
        let pyval: &Bound<'_, PyVal> = item.downcast()?;
        let pyval = pyval.borrow();
        match &*pyval {
            PyVal::Sym(var_id) => names.push(*var_id),
            _ => panic!("Invalid input variable: {:?}", pyval),
        }
    }

    Ok(Wire::from_iter(
        names.into_iter().map(|val| (val, "tensor")),
    ))
}

#[pymethods]
impl PyAtom {
    #[new]
    fn new(
        ctx: &Bound<'_, Context>,
        read: &Bound<'_, PyList>,
        write: &Bound<'_, PyList>,
        init: &Bound<'_, PyList>,
        update: &Bound<'_, PyList>,
    ) -> Self {
        let read = vars_to_wiring(read).unwrap();
        let write = vars_to_wiring(write).unwrap();

        let ctx: &mut Context = &mut ctx.borrow_mut();
        let init = pyterms_to_torchterms(ctx, init).unwrap();
        let update = pyterms_to_torchterms(ctx, update).unwrap();

        let atom = Atom::new_unchecked(read, write, Wire::none(), init, update);
        Self { atom }
    }

    fn dbg(&self) {
        println!("{:?}", self);
    }
}

impl std::fmt::Debug for PyAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Reads: {:?}", self.atom.read())?;
        writeln!(f, "Writes: {:?}", self.atom.ctrl())?;
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
