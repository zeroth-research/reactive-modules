use crate::dtype::DType;
use crate::itype::{IType, Val, ArithOp, LogicalOp, CmpOp};

use base::Term;
use base::module::Module;

pub fn parse_modules(modules: &[Module<DType, IType>]) -> String {
    let mut out = String::new();

    for module in modules {
        out.push_str(&format!("; ========================================\n; MODULE {}\n; ========================================\n", module.name().unwrap_or("<unnamed>")));

        out.push_str(&format!("\n; =====================\n; = Wire Declarations =\n; =====================\n"));

        let pair = module.wire();
        let wires_ltc = &pair[0];
        let wires_nxt = &pair[1];

        let offset = wires_ltc.len();

        out.push_str("\n; Latched Wires:\n");
        for (id, dtype) in wires_ltc.iter() {
            let name = wire_name(id, offset);
            out.push_str(&format!("(declare-fun {} () {})\n", name, dtype));
        }

        out.push_str("\n; Next Wires:\n");
        for (id, dtype) in wires_nxt.iter() {
            let name = wire_name(id, offset);
            out.push_str(&format!("(declare-fun {} () {})\n", name, dtype));
        }

        for (atom_index, atom) in module.atoms().iter().enumerate() {
            out.push_str(&format!("\n; ==============================\n; ATOM {}:\n; ==============================\n", atom_index));
            
            // Get the output wires (what this atom controls)
            let output_wires: Vec<usize> = atom.ctrl().iter().map(|(w, _)| w).collect();

            let init = atom.init();
            let update = atom.update();
            
            out.push_str("\n; ==============\n; = Init Terms =\n; ==============\n\n");
            for &output_wire in &output_wires {
                let expr = smt_trace_expr(output_wire, init, offset);
                let wire = wire_name(output_wire, offset);
                out.push_str(&format!("(assert (= {} {}))\n", wire, expr));
            }

            out.push_str("\n; ================\n; = Update Terms =\n; ================\n\n");
            for &output_wire in &output_wires {
                let expr = smt_trace_expr(output_wire, update, offset);
                let wire = wire_name(output_wire, offset);
                out.push_str(&format!("(assert (= {} {}))\n", wire, expr));
            }
        }
    }
    out
}

fn wire_name(id: usize, offset: usize) -> String {
    if id >= offset {
        format!("w{}_{}", id - offset, 1)
    } else {
        format!("w{}_{}", id, 0)
    }
}

fn smt_trace_expr(wire_id: usize, terms: &[Term<DType, IType>], offset: usize) -> String {
    // Find the term that writes to this wire
    let term = terms.iter().find(|t| t.writes().iter().any(|(w, _)| w == wire_id));
    
    match term {
        None => {
            // No term writes this wire: it's an input
            wire_name(wire_id, offset)
        }
        Some(term) => {
            // Some term writes this wire: recursively trace it
            match term.itype() {
                IType::Const(val) => match val {
                    Val::Real(x) => x.to_string(),
                    Val::Int(x) => x.to_string(),
                    Val::Bool(b) => b.to_string(),
                    Val::None => panic!("Cannot emit None"),
                },
                IType::Arith(op) => {
                    let input1 = term.reads().iter().nth(0).unwrap().0;
                    let input2 = term.reads().iter().nth(1).unwrap().0;
                    
                    let expr1 = smt_trace_expr(input1, terms, offset);
                    let expr2 = smt_trace_expr(input2, terms, offset);
                    
                    match op {
                        ArithOp::Add => format!("(+ {} {})", expr1, expr2),
                        ArithOp::Sub => format!("(- {} {})", expr1, expr2),
                        ArithOp::Mul => format!("(* {} {})", expr1, expr2),
                        ArithOp::Div => format!("(/ {} {})", expr1, expr2),
                    }
                }
                IType::Logical(op) => {
                    match op {
                        LogicalOp::Not => {
                            let input = term.reads().iter().nth(0).unwrap().0;
                            let expr = smt_trace_expr(input, terms, offset);
                            return format!("(not {})", expr);
                        }
                        LogicalOp::And | LogicalOp::Or => {
                            let input1 = term.reads().iter().nth(0).unwrap().0;
                            let input2 = term.reads().iter().nth(1).unwrap().0;
                            
                            let expr1 = smt_trace_expr(input1, terms, offset);
                            let expr2 = smt_trace_expr(input2, terms, offset);

                            match op {
                                LogicalOp::And => format!("(and {} {})", expr1, expr2),
                                LogicalOp::Or  => format!("(or {} {})", expr1, expr2),
                                _ => unreachable!(),
                            }
                        }
                    }
                }
                IType::Cmp(op) => {
                    let input1 = term.reads().iter().nth(0).unwrap().0;
                    let input2 = term.reads().iter().nth(1).unwrap().0;

                    let expr1 = smt_trace_expr(input1, terms, offset);
                    let expr2 = smt_trace_expr(input2, terms, offset);

                    match op {
                        CmpOp::Eq => format!("(= {} {})", expr1, expr2),
                        CmpOp::Lt => format!("(< {} {})", expr1, expr2),
                        CmpOp::Le => format!("(<= {} {})", expr1, expr2),
                        CmpOp::Gt => format!("(> {} {})", expr1, expr2),
                        CmpOp::Ge => format!("(>= {} {})", expr1, expr2),
                    }
                }
                IType::Id => {
                    let input_id = term.reads().iter().nth(0).unwrap().0;
                    smt_trace_expr(input_id, terms, offset)
                }
                IType::Cond => {
                    let cond_id = term.reads().iter().nth(0).unwrap().0;
                    let then_id = term.reads().iter().nth(1).unwrap().0;
                    let else_id = term.reads().iter().nth(2).unwrap().0;

                    let cond_expr = smt_trace_expr(cond_id, terms, offset);
                    let then_expr = smt_trace_expr(then_id, terms, offset);
                    let else_expr = smt_trace_expr(else_id, terms, offset);

                    format!("(ite {} {} {})", cond_expr, then_expr, else_expr)
                }
            }
        }
    }
}
