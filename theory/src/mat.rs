/*!
# Matrices

A Matrix type is parametrized by another [Type]
and its dimensions `M` and `N` which are `usize`.

This code makes available: TODO

## Examples
TODO

*/

use std::marker::PhantomData;

use crate::*;

/// Bitvector type parametrized by its size
#[derive(Clone)]
pub struct Mat<T: Type, const M: usize, const N: usize>(pub std::marker::PhantomData<T>);
impl<T: Type, const M: usize, const N: usize> Type for Mat<T, M, N> {}

#[derive(Copy, Clone)]
pub struct MatMul<T: Type, const A: usize, const B: usize, const C: usize> {
    pub t: PhantomData<T>,
}
impl<T: Type + Copy, const A: usize, const B: usize, const C: usize> Operation
    for MatMul<T, A, B, C>
{
}

impl<T: Type + Copy, const A: usize, const B: usize, const C: usize>
    Operation2To1<Mat<T, A, B>, Mat<T, B, C>, Mat<T, A, C>> for MatMul<T, A, B, C>
{
}

mk_theory!(
    Types([T: Type, const M: usize, const N: usize] Mat => Mat<T, M, N>),
    {
        [T: Type, const M: usize, const N: usize]
        Add(Mat<T, M, N>, Mat<T, M, N>) => Mat<T, M, N>,
        Id(Mat<T, M, N>) => Mat<T, M, N>
    }
    {
        [T: Type + Copy, const A: usize, const B: usize, const C: usize]
        MatMul => MatMul<T, A, B, C>
    }
);
