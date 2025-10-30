// TODO: remove in the future
#![allow(dead_code)]

use std::fmt;

pub mod atom;
pub mod module;
pub mod term;
pub mod wire;

fn write_indented<W: fmt::Write, I: fmt::Display>(f: &mut W, prefix: &str, item: I) -> fmt::Result {
    let s = format!("{}", item);
    let mut lines = s.lines();

    if let Some(first_line) = lines.next() {
        writeln!(f, "{}{}", prefix, first_line)?;

        for line in lines {
            writeln!(f, "{}{}", prefix, line)?;
        }
    }

    Ok(())
}
