use std::collections::HashSet;

use crate::dtype::DType;
use crate::itype::{ArithOp, CmpOp, IType, LogicalOp, Val};

use base::{Atom, Module, Term};

type SmtAtom = Atom<DType, IType>;
type SmtModule = Module<DType, IType>;

pub struct AtomSmtLibTranslator<'a>(&'a SmtAtom);

impl AtomSmtLibTranslator<'_> {
    pub fn ctrl(&self) -> Vec<String> {
        self.0
            .ctrl()
            .wires()
            .map(|w| declare_var(w.id(), w.dtype()))
            .collect::<Vec<String>>()
    }

    pub fn read(&self) -> Vec<String> {
        self.0
            .read()
            .wires()
            .map(|w| declare_var(w.id(), w.dtype()))
            .collect::<Vec<String>>()
    }

    pub fn wait(&self) -> Vec<String> {
        self.0
            .wait()
            .wires()
            .map(|w| declare_var(w.id(), w.dtype()))
            .collect::<Vec<String>>()
    }

    // Temporary wires
    pub fn temp(&self) -> Vec<String> {
        let mut ret: Vec<String> = Vec::new();

        // temporary are those variables that are written to but are not controlled
        // (these could be also private, but we do not have private variables atm.)
        let ctrl: HashSet<usize> = HashSet::from_iter(self.0.ctrl().ids());
        for term in self.0.init().iter().chain(self.0.update().iter()) {
            for wire in term.write().wires() {
                if !ctrl.contains(&wire.id()) {
                    ret.push(declare_var(wire.id(), wire.dtype()))
                }
            }
        }

        ret
    }

    // return a list of smtlib assertions (as String) that represent the init terms
    pub fn init(&self) -> Vec<String> {
        self.0
            .init()
            .iter()
            .map(|term| {
                let write_id = term.write().ids().next().unwrap();
                format!("(assert (= {} {}))", wire_name(write_id), smt_expr(term))
            })
            .collect()
    }

    // return a list of smtlib assertions (as String) that represent the update terms
    pub fn update(&self) -> Vec<String> {
        self.0
            .update()
            .iter()
            .map(|term| {
                let write_id = term.write().ids().next().unwrap();
                format!("(assert (= {} {}))", wire_name(write_id), smt_expr(term))
            })
            .collect()
    }

    fn to_smtlib(&self) -> String {
        format!(
            ";;; Atom\n\n;; Controls\n{}\n\n;; Reads\n{}\n\n;; Awaits\n{}\n\n;; Temporary\n{}\n\n;; Init\n{}\n\n;; Update\n{}",
            self.ctrl().join("\n"),
            self.read().join("\n"),
            self.wait().join("\n"),
            self.temp().join("\n"),
            self.init().join("\n"),
            self.update().join("\n"),
        )
    }

    fn body_to_smtlib(&self) -> String {
        format!(
            ";;; Atom\n\n;; Temporary\n{}\n\n;; Init\n{}\n\n;; Update\n{}",
            self.temp().join("\n"),
            self.init().join("\n"),
            self.update().join("\n"),
        )
    }
}

pub struct ModuleSmtLibTranslator<'a>(&'a SmtModule);

impl ModuleSmtLibTranslator<'_> {
    /// return smtlib declarations of `intf` variables
    fn intf(&self) -> Vec<String> {
        let vars = self.0.intf();
        vars[0]
            .iter()
            .chain(vars[1].iter())
            .map(|w| declare_var(w.id(), w.dtype()))
            .collect::<Vec<String>>()
    }

    /// return smtlib declarations of `extl` variables
    fn extl(&self) -> Vec<String> {
        let vars = self.0.extl();
        vars[0]
            .iter()
            .chain(vars[1].iter())
            .map(|w| declare_var(w.id(), w.dtype()))
            .collect::<Vec<String>>()
    }

    fn to_smtlib(&self) -> String {
        // we do not consider private variables atm.
        // debug_assert!(self.0.prvt().is_empty());

        format!(
            ";;; Module\n\n;; Interface\n{}\n\n;; External\n{}\n\n;; Atoms\n\n{}",
            self.intf().join("\n"),
            self.extl().join("\n"),
            self.0
                .atoms()
                .iter()
                .map(|atom| AtomSmtLibTranslator(atom).body_to_smtlib())
                .collect::<Vec<String>>()
                .join("\n")
        )
    }
}

pub fn module_to_smtlib(module: &Module<DType, IType>) -> String {
    ModuleSmtLibTranslator(module).to_smtlib()
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
        IType::Const(val) => match val {
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
