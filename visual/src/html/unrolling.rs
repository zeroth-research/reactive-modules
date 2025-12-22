use base::Term;

use common::transition::{Transition, WiredTransitions};

pub fn write_to_html<T, I>(
    module: &WiredTransitions<T, I>,
    path: &str,
) -> Result<(), std::io::Error>
where
    Term<T, I>: std::fmt::Display,
    I: std::fmt::Display,
    T: std::fmt::Display,
{
    todo!()
}
