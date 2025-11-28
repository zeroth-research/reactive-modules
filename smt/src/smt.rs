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

        out.push_str("\n; Latched Wires:\n");
        for (id, dtype) in wires_ltc.iter() {
            let name = wire_name(id);
            out.push_str(&format!("(declare-fun {} () {})\n", name, dtype));
        }

        out.push_str("\n; Next Wires:\n");
        for (id, dtype) in wires_nxt.iter() {
            let name = wire_name(id);
            out.push_str(&format!("(declare-fun {} () {})\n", name, dtype));
        }

        let mut temp_wires: std::collections::HashMap<usize, DType> = std::collections::HashMap::new();

        for atom in module.atoms() {
            for term in atom.init().iter().chain(atom.update().iter()) {
                for (wire_id, dtype) in term.writes().iter() {
                    if !wires_ltc.iter().any(|(id, _)| id == wire_id) &&
                    !wires_nxt.iter().any(|(id, _)| id == wire_id) {
                        temp_wires.insert(wire_id, *dtype);
                    }
                }
            }
        }

        if !temp_wires.is_empty() {
            out.push_str("\n; Temporary Wires:\n");
            let mut temp_vec: Vec<_> = temp_wires.into_iter().collect();
            temp_vec.sort_by_key(|(id, _)| *id);
            
            for (id, dtype) in temp_vec {
                out.push_str(&format!("(declare-fun {} () {})\n", wire_name(id), dtype));
            }
        }

        for (atom_index, atom) in module.atoms().iter().enumerate() {
            out.push_str(&format!("\n; ==============================\n; ATOM {}:\n; ==============================\n", atom_index));

            let init = atom.init();
            let update = atom.update();
            
            out.push_str("\n; ==============\n; = Init Terms =\n; ==============\n\n");
            for term in init {
                let (write_id, _) = term.writes().iter().next().unwrap();
                let expr = smt_expr(term);
                out.push_str(&format!("(assert (= {} {}))\n", wire_name(write_id), expr));
            }

            out.push_str("\n; ================\n; = Update Terms =\n; ================\n\n");
            for term in update {
                let (write_id, _) = term.writes().iter().next().unwrap();
                let expr = smt_expr(term);
                out.push_str(&format!("(assert (= {} {}))\n", wire_name(write_id), expr));
            }
        }
    }
    out
}

fn wire_name(id: usize) -> String {
    format!("w{}", id)
}

fn smt_expr(term: &Term<DType, IType>) -> String {
    match term.itype() {
        IType::Const(val) => match val {
            Val::Real(x) => x.to_string(),
            Val::Int(x) => x.to_string(),
            Val::Bool(b) => b.to_string(),
            Val::None => panic!("Cannot emit None"),
        },
        
        IType::Arith(op) => {
            let args: Vec<String> = term.reads().iter()
                .map(|(id, _)| wire_name(id))
                .collect();
            
            match op {
                ArithOp::Add => format!("(+ {} {})", args[0], args[1]),
                ArithOp::Sub => format!("(- {} {})", args[0], args[1]),
                ArithOp::Mul => format!("(* {} {})", args[0], args[1]),
                ArithOp::Div => format!("(/ {} {})", args[0], args[1]),
            }
        }
        
        IType::Logical(op) => {
            let args: Vec<String> = term.reads().iter()
                .map(|(id, _)| wire_name(id))
                .collect();
            
            match op {
                LogicalOp::Not => format!("(not {})", args[0]),
                LogicalOp::And => format!("(and {} {})", args[0], args[1]),
                LogicalOp::Or  => format!("(or {} {})", args[0], args[1]),
            }
        }
        
        IType::Cmp(op) => {
            let args: Vec<String> = term.reads().iter()
                .map(|(id, _)| wire_name(id))
                .collect();
            
            match op {
                CmpOp::Eq => format!("(= {} {})", args[0], args[1]),
                CmpOp::Lt => format!("(< {} {})", args[0], args[1]),
                CmpOp::Le => format!("(<= {} {})", args[0], args[1]),
                CmpOp::Gt => format!("(> {} {})", args[0], args[1]),
                CmpOp::Ge => format!("(>= {} {})", args[0], args[1]),
            }
        }
        
        IType::Id => {
            let (id, _) = term.reads().iter().next().unwrap();
            wire_name(id)
        }
        
        IType::Cond => {
            let c = wire_name(term.reads().iter().nth(0).unwrap().0);
            let t = wire_name(term.reads().iter().nth(1).unwrap().0);
            let e = wire_name(term.reads().iter().nth(2).unwrap().0);
            format!("(ite {} {} {})", c, t, e)
        }
    }
}
