use crate::types::IType;
use tch::IndexOp;
use tch::Tensor;

pub fn eval(op: &IType, read: &[&Tensor]) -> Result<Vec<Tensor>, String> {
    match op {
        IType::Tensor(t) => {
            debug_assert!(read.is_empty());
            Ok(vec![t.tensor.shallow_clone()])
        }
        IType::Id() => {
            debug_assert!(read.len() == 1);
            Ok(vec![read[0].shallow_clone()])
        }
        // Arithmetic
        IType::Add() => {
            debug_assert!(read.len() == 2);
            Ok(vec![read[0] + read[1]])
        }
        IType::Sub() => {
            debug_assert!(read.len() == 2);
            Ok(vec![read[0] - read[1]])
        }
        IType::Mul() => {
            debug_assert!(read.len() == 2);
            Ok(vec![read[0] * read[1]])
        }
        IType::Div() => {
            debug_assert!(read.len() == 2);
            Ok(vec![read[0] / read[1]])
        }
        IType::MatMul() => {
            debug_assert!(read.len() == 2);
            Ok(vec![read[0].matmul(read[1])])
        }
        // Comparisons
        IType::Eq() => {
            debug_assert!(read.len() == 2);
            Ok(vec![read[0].eq_tensor(read[1])])
        }
        IType::Neq() => {
            debug_assert!(read.len() == 2);
            Ok(vec![read[0].ne_tensor(read[1])])
        }
        IType::Lt() => {
            debug_assert!(read.len() == 2);
            Ok(vec![read[0].less_tensor(read[1])])
        }
        IType::Le() => {
            debug_assert!(read.len() == 2);
            Ok(vec![read[0].le_tensor(read[1])])
        }
        IType::Gt() => {
            debug_assert!(read.len() == 2);
            Ok(vec![read[0].greater_tensor(read[1])])
        }
        IType::Ge() => {
            debug_assert!(read.len() == 2);
            Ok(vec![read[0].ge_tensor(read[1])])
        }
        // Logical
        IType::And() => {
            debug_assert!(read.len() == 2);
            Ok(vec![read[0].logical_and(read[1])])
        }
        IType::Or() => {
            debug_assert!(read.len() == 2);
            Ok(vec![read[0].logical_or(read[1])])
        }
        IType::Not() => {
            debug_assert!(read.len() == 1);
            Ok(vec![read[0].logical_not()])
        }
        // Control flow
        IType::Ite() => {
            debug_assert!(read.len() == 3);
            // where_self(condition, other): selects from self where cond is true, other where false
            Ok(vec![read[1].where_self(read[0], read[2])])
        }
        // Special
        IType::Argmax() => {
            debug_assert!(read.len() == 1);
            Ok(vec![read[0].argmax(None, false)])
        }
        IType::ReLU() => {
            debug_assert!(read.len() == 1);
            Ok(vec![read[0].relu()])
        }
        // Tensor reductions
        IType::TensorSum() => {
            debug_assert!(read.len() == 1);
            Ok(vec![read[0].sum(tch::Kind::Float)])
        }
        IType::TensorMean() => {
            debug_assert!(read.len() == 1);
            Ok(vec![read[0].mean(tch::Kind::Float)])
        }
        IType::TensorMax() => {
            debug_assert!(read.len() == 1);
            Ok(vec![read[0].max()])
        }
        // Tensor indexing
        IType::TensorGet() => {
            debug_assert!(read.len() == 2);
            let idx = i64::try_from(read[1]).map_err(|e| format!("TensorGet index: {e}"))?;
            let flat = read[0].view([-1]);
            Ok(vec![flat.i(idx)])
        }
        IType::TensorSet() => {
            debug_assert!(read.len() == 3);
            // read[0] = tensor, read[1] = index, read[2] = value
            let idx = i64::try_from(read[1]).map_err(|e| format!("TensorSet index: {e}"))?;
            let result = read[0].shallow_clone();
            let flat = result.view([-1]);
            let _ = flat.i(idx).copy_(read[2]);
            Ok(vec![result])
        }
        IType::Uninterpreted(name) => {
            Err(format!("cannot evaluate uninterpreted function '{name}'"))
        }
    }
}
