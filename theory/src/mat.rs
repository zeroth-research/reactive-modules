/*!
# Matrices with parametric element type
  and shape given in runtime.

TODO: add description

## Examples

```
// TODO: add examples
```
*/

use crate::*;

#[derive(Copy, Clone, PartialEq)]
pub enum MatDType<T: PartialEq> {
    Mat(usize, usize),
    _Phantom(std::marker::PhantomData<T>),
}

/// TODO: write "formal" semantics
#[derive(Clone, PartialEq)]
pub enum Mat<T> {
    // TODO: use bitarray
    Const(Vec<Vec<T>>),
    Add,
    Mul,
    MatMul,
    Prod,
    Sum,
    Id,
}

impl<T: PartialEq> Theory for Mat<T> {
    type DType = MatDType<T>;

    fn _check(&self, read: &[Self::DType], write: &[Self::DType]) -> bool {
        match self {
            Mat::Const(cm) => {
                read.len() == 0
                    && write.len() == 1
                    && match &write[0] {
                        MatDType::Mat(i, j) => {
                            cm.len() == *i && cm.iter().all(|row| row.len() == *j)
                        }
                        _ => false,
                    }
            }
            Mat::Prod | Mat::Sum => {
                read.len() == 1
                    && write.len() == 1
                    && match write[0] {
                        MatDType::Mat(1, 1) => true,
                        _ => false,
                    }
            }
            Mat::Id => read.len() == 1 && write.len() == 1 && read[0] == write[0],
            Mat::Add | Mat::Mul => read.len() == 2 && write.len() == 1 && read[0] == write[0],
            Mat::MatMul => {
                read.len() == 2
                    && write.len() == 1
                    && match (&read[0], &read[1], &write[0]) {
                        (MatDType::Mat(d1, d2), MatDType::Mat(d3, d4), MatDType::Mat(d5, d6)) => {
                            d2 == d3 && d1 == d5 && d4 == d6
                        }
                        _ => false,
                    }
            }
        }
    }
}
