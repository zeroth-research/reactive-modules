use crate::primitives::variable::Variable;

pub trait Action<V, S>
where
    V: Variable,
    for<'a> &'a S: IntoIterator<Item = &'a V>,
    for<'a> <&'a S as IntoIterator>::IntoIter: ExactSizeIterator,
{
    fn read(&self) -> &S;
    fn write(&self) -> &S;
}
