use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::dtype::DType;
use crate::itype::{ArithOp, CmpOp, IType, LogicalOp, Val};

use base::Term;

/// Errors that can occur during SMT-LIB assertion generation.
#[derive(Debug)]
pub enum SmtError {
    /// A term reads a wire that is written by a later term (bad topological order).
    BadOrder { wire_id: usize, read_by_term: usize, written_by_term: usize },
    /// Two terms write the same wire.
    DuplicateWrite { wire_id: usize, first_term: usize, second_term: usize },
    /// A formatting error occurred while writing output.
    Fmt(fmt::Error),
}

impl fmt::Display for SmtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtError::BadOrder { wire_id, read_by_term, written_by_term } => {
                write!(
                    f,
                    "bad order: wire w{} is read by term {} but written by later term {}",
                    wire_id, read_by_term, written_by_term
                )
            }
            SmtError::DuplicateWrite { wire_id, first_term, second_term } => {
                write!(
                    f,
                    "duplicate write: wire w{} is written by both term {} and term {}",
                    wire_id, first_term, second_term
                )
            }
            SmtError::Fmt(e) => write!(f, "format error: {}", e),
        }
    }
}

impl From<fmt::Error> for SmtError {
    fn from(e: fmt::Error) -> Self {
        SmtError::Fmt(e)
    }
}

/// Write `(declare-fun wi () Sort)` for each wire.
pub fn declare<'a>(
    wires: impl IntoIterator<Item = &'a base::Wire<DType>>,
    w: &mut impl fmt::Write,
) -> fmt::Result {
    for wire in wires {
        writeln!(w, "(declare-fun {} () {})", wire_name(wire.id()), wire.dtype())?;
    }
    Ok(())
}

/// Write an SMT-LIB assertion for a sequence of terms (a closed diagram).
///
/// Intermediate wires (written by one term, read by another) become `let`-bindings.
/// Output wires (written but not read by any term) become equalities inside the assertion.
///
/// # Errors
///
/// Returns `SmtError::BadOrder` if a term reads a wire written by a later term,
/// `SmtError::DuplicateWrite` if two terms write the same wire.
pub fn assert_terms<'a>(
    terms: impl IntoIterator<Item = &'a Term<DType, IType>>,
    w: &mut impl fmt::Write,
) -> Result<(), SmtError> {
    let terms: Vec<&Term<DType, IType>> = terms.into_iter().collect();
    if terms.is_empty() {
        return Ok(());
    }

    // Pass 1a: collect all writes and check for duplicates
    let mut written_by: HashMap<usize, usize> = HashMap::new(); // wire_id -> term_index
    for (i, term) in terms.iter().enumerate() {
        for wire_id in term.write().ids() {
            if let Some(&prev) = written_by.get(&wire_id) {
                return Err(SmtError::DuplicateWrite {
                    wire_id,
                    first_term: prev,
                    second_term: i,
                });
            }
            written_by.insert(wire_id, i);
        }
    }

    // Pass 1b: collect reads and validate ordering
    let mut read_set: HashSet<usize> = HashSet::new();
    for (i, term) in terms.iter().enumerate() {
        for wire_id in term.read().ids() {
            read_set.insert(wire_id);
            if let Some(&writer) = written_by.get(&wire_id)
                && writer >= i {
                    return Err(SmtError::BadOrder {
                        wire_id,
                        read_by_term: i,
                        written_by_term: writer,
                    });
                }
        }
    }

    // Pass 2: classify and emit
    // Intermediate: written AND read by another term -> let-binding
    // Output: written but NOT read by any term -> equality
    let mut let_bindings = Vec::new();
    let mut equalities = Vec::new();

    for term in &terms {
        let wire_id = term.write().ids().next().unwrap();
        let expr = smt_expr(term);

        if read_set.contains(&wire_id) {
            let_bindings.push(format!("({} {})", wire_name(wire_id), expr));
        } else {
            equalities.push(format!("(= {} {})", wire_name(wire_id), expr));
        }
    }

    if let_bindings.is_empty() {
        for eq in &equalities {
            writeln!(w, "(assert {})", eq)?;
        }
    } else {
        write!(w, "(assert\n  (let (")?;
        for (i, binding) in let_bindings.iter().enumerate() {
            if i > 0 {
                write!(w, "\n        ")?;
            }
            write!(w, "{}", binding)?;
        }
        write!(w, ")\n    (and ")?;
        for (i, eq) in equalities.iter().enumerate() {
            if i > 0 {
                write!(w, "\n         ")?;
            }
            write!(w, "{}", eq)?;
        }
        writeln!(w, ")))")?;
    }

    Ok(())
}

fn wire_name(id: usize) -> String {
    format!("w{}", id)
}

fn smt_expr(term: &Term<DType, IType>) -> String {
    match term.itype() {
        IType::Num(val) => match val {
            Val::Real(x) => x.to_string(),
            Val::Int(x) => x.to_string(),
            Val::Bool(b) => b.to_string(),
            Val::None => panic!("Cannot emit None"),
        },

        IType::Arith(op) => {
            let args: Vec<String> = term.read().wires().map(|w| wire_name(w.id())).collect();

            match op {
                ArithOp::Add => format!("(+ {} {})", args[0], args[1]),
                ArithOp::Sub => format!("(- {} {})", args[0], args[1]),
                ArithOp::Mul => format!("(* {} {})", args[0], args[1]),
                ArithOp::Div => format!("(/ {} {})", args[0], args[1]),
            }
        }

        IType::Logical(op) => {
            let args: Vec<String> = term.read().wires().map(|w| wire_name(w.id())).collect();

            match op {
                LogicalOp::Not => format!("(not {})", args[0]),
                LogicalOp::And => format!("(and {} {})", args[0], args[1]),
                LogicalOp::Or => format!("(or {} {})", args[0], args[1]),
            }
        }

        IType::Cmp(op) => {
            let args: Vec<String> = term.read().wires().map(|w| wire_name(w.id())).collect();

            match op {
                CmpOp::Eq => format!("(= {} {})", args[0], args[1]),
                CmpOp::Lt => format!("(< {} {})", args[0], args[1]),
                CmpOp::Le => format!("(<= {} {})", args[0], args[1]),
                CmpOp::Gt => format!("(> {} {})", args[0], args[1]),
                CmpOp::Ge => format!("(>= {} {})", args[0], args[1]),
            }
        }

        IType::Sin => {
            let arg = wire_name(term.read().wires().next().unwrap().id());
            format!("(sin {})", arg)
        }

        IType::Cos => {
            let arg = wire_name(term.read().wires().next().unwrap().id());
            format!("(cos {})", arg)
        }

        IType::Id => {
            let id = term.read().wires().next().unwrap().id();
            wire_name(id)
        }

        IType::Cond => {
            let c = wire_name(term.read().wires().next().unwrap().id());
            let t = wire_name(term.read().wires().nth(1).unwrap().id());
            let e = wire_name(term.read().wires().nth(2).unwrap().id());
            format!("(ite {} {} {})", c, t, e)
        }
    }
}
