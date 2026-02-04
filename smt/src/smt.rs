use std::collections::HashSet;

use crate::dtype::DType;
use crate::itype::{ArithOp, CmpOp, IType, LogicalOp, Val};

use base::{Atom, Module, Term};

type SmtAtom = Atom<DType, IType>;
type SmtModule = Module<DType, IType>;

pub struct AtomSmtLibTranslator<'a>(&'a SmtAtom);

pub fn wires_decls<'a, T: IntoIterator<Item = &'a base::Wire<DType>>>(wires: T) -> Vec<String> {
    wires
        .into_iter()
        .map(|w| declare_var(w.id(), w.dtype()))
        .collect::<Vec<String>>()
}

impl AtomSmtLibTranslator<'_> {
    fn method_to_smtlib<'a, T>(&self, terms: T, state_wires: &HashSet<usize>) -> String 
    where T: IntoIterator<Item = &'a Term<DType, IType>>
    {
        let mut let_bindings = Vec::new();
        let mut state_assigns = Vec::new();

        for term in terms {
            let wire_id = term.write().ids().next().unwrap();
            let expr = smt_expr(term);
            
            if state_wires.contains(&wire_id) {
                state_assigns.push(format!("(= {} {})", wire_name(wire_id), expr));
            } else {
                let_bindings.push(format!("({} {})", wire_name(wire_id), expr));
            }
        }

        if let_bindings.is_empty() {
            state_assigns.iter().map(|s| format!("(assert {})", s)).collect::<Vec<_>>().join("\n")
        } else {
            format!(
                "(assert\n  (let ({})\n    (and {})))",
                let_bindings.join("\n        "),
                state_assigns.join("\n         ")
            )
        }
    }

    pub fn init(&self, state_wires: &HashSet<usize>) -> String {
        self.method_to_smtlib(self.0.init().iter(), state_wires)
    }

    pub fn update(&self, state_wires: &HashSet<usize>) -> String {
        self.method_to_smtlib(self.0.update().iter(), state_wires)
    }
}

pub struct ModuleSmtLibTranslator<'a>(&'a SmtModule);

impl ModuleSmtLibTranslator<'_> {
    fn state_wire_ids(&self) -> HashSet<usize> {
        let mut ids = HashSet::new();
        // Collect intf, extl, prvt wire IDs (both latched and next)
        for wire in self.0.intf().latched().iter().chain(self.0.intf().next().iter()) {
            ids.insert(wire.id());
        }
        for wire in self.0.extl().latched().iter().chain(self.0.extl().next().iter()) {
            ids.insert(wire.id());
        }
        for wire in self.0.prvt().latched().iter().chain(self.0.prvt().next().iter()) {
            ids.insert(wire.id());
        }
        ids
    }
    /// return smtlib declarations of `intf` variables
    pub fn intf(&self) -> Vec<String> {
        let vars = self.0.intf();
        vars.latched()
            .iter()
            .chain(vars.next().iter())
            .map(|w| declare_var(w.id(), w.dtype()))
            .collect::<Vec<String>>()
    }

    /// return smtlib declarations of `extl` variables
    pub fn extl(&self) -> Vec<String> {
        let vars = self.0.extl();
        vars.latched()
            .iter()
            .chain(vars.next().iter())
            .map(|w| declare_var(w.id(), w.dtype()))
            .collect::<Vec<String>>()
    }

    pub fn variables_to_smtlib(&self) -> String {
        format!(
            ";; Interface\n{}\n\n;; External\n{}",
            self.intf().join("\n"),
            self.extl().join("\n"),
        )
    }

    pub fn to_smtlib(&self) -> String {
        if !self.0.prvt().is_empty() {
            unimplemented!()
        }

        let state_wires = self.state_wire_ids();
        
        format!(
            ";;; Module\n\n{}\n\n;; Atoms\n{}",
            self.variables_to_smtlib(),
            self.0
                .atoms()
                .iter()
                .enumerate()
                .map(|(n, atom)| {
                    let translator = AtomSmtLibTranslator(atom);
                    format!(
                        "\n;; --- Atom {} ---\n;; Init\n{}\n\n;; Update\n{}",
                        n,
                        translator.init(&state_wires),
                        translator.update(&state_wires)
                    )
                })
                .collect::<Vec<String>>()
                .join("\n")
        )
    }

    pub fn init_to_smtlib(&self) -> String {
        let state_wires = self.state_wire_ids();
        self.0
            .atoms()
            .iter()
            .enumerate()
            .map(|(n, atom)| {
                format!(
                    "\n;; --- Atom {} ---\n{}",
                    n,
                    AtomSmtLibTranslator(atom).init(&state_wires)
                )
            })
            .collect::<Vec<String>>()
            .join("\n")
    }

    pub fn update_to_smtlib(&self) -> String {
        let state_wires = self.state_wire_ids();
        self.0
            .atoms()
            .iter()
            .enumerate()
            .map(|(n, atom)| {
                format!(
                    "\n;; --- Atom {} ---\n{}",
                    n,
                    AtomSmtLibTranslator(atom).update(&state_wires)
                )
            })
            .collect::<Vec<String>>()
            .join("\n")
    }
}

pub fn module_to_smtlib(module: &Module<DType, IType>) -> String {
    ModuleSmtLibTranslator(module).to_smtlib()
}

pub fn module_init_to_smtlib(module: &Module<DType, IType>) -> String {
    ModuleSmtLibTranslator(module).init_to_smtlib()
}

pub fn module_update_to_smtlib(module: &Module<DType, IType>) -> String {
    ModuleSmtLibTranslator(module).update_to_smtlib()
}

pub fn module_variables_to_smtlib(module: &Module<DType, IType>) -> String {
    ModuleSmtLibTranslator(module).variables_to_smtlib()
}

pub fn parse_modules(modules: &[Module<DType, IType>]) -> String {
    modules
        .iter()
        .map(module_to_smtlib)
        .collect::<Vec<String>>()
        .join("\n")
}

fn wire_name(id: usize) -> String {
    format!("w{}", id)
}

fn declare_var(id: usize, ty: &DType) -> String {
    format!("(declare-fun {} () {})", wire_name(id), ty)
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
