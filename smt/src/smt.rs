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
    pub fn ctrl(&self) -> Vec<String> {
        wires_decls(self.0.ctrl().wires())
    }

    pub fn read(&self) -> Vec<String> {
        wires_decls(self.0.read().wires())
    }

    pub fn wait(&self) -> Vec<String> {
        wires_decls(self.0.wait().wires())
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

    /// Declare variables that are defined in this atom, i.e., temporary variables
    /// (all others are declared on the level of the module)
    pub fn temps_to_smtlib(&self) -> String {
        format!(";;; Temporary\n{}", self.temp().join("\n"),)
    }

    pub fn methods_to_smtlib(&self) -> String {
        format!(
            ";; Init\n{}\n\n;; Update\n{}",
            self.init().join("\n"),
            self.update().join("\n"),
        )
    }

    pub fn to_smtlib(&self) -> String {
        format!(
            ";;; {}\n\n;;\n\n{}",
            self.temps_to_smtlib(),
            self.methods_to_smtlib(),
        )
    }

    pub fn to_smtlib_full(&self) -> String {
        format!(
            ";;; Atom\n\n;; Controls\n{}\n\n;; Reads\n{}\n\n;; Awaits\n{}\n\n{}",
            self.ctrl().join("\n"),
            self.read().join("\n"),
            self.wait().join("\n"),
            self.to_smtlib(),
        )
    }
}

pub struct ModuleSmtLibTranslator<'a>(&'a SmtModule);

impl ModuleSmtLibTranslator<'_> {
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

    /// return smtlib declarations of `temp` variables from all atoms
    pub fn temp(&self) -> Vec<String> {
        wires_decls(self.0.temp())
    }

    pub fn observable_variables_to_smtlib(&self) -> String {
        format!(
            ";; Interface\n{}\n\n;; External\n{}",
            self.intf().join("\n"),
            self.extl().join("\n"),
        )
    }

    pub fn variables_to_smtlib(&self) -> String {
        format!(
            "{}\n\n;; Temporary\n{}",
            self.observable_variables_to_smtlib(),
            self.temp().join("\n"),
        )
    }

    pub fn to_smtlib(&self) -> String {
        // we do not consider private variables atm.
        if !self.0.prvt().is_empty() {
            unimplemented!()
        }

        format!(
            ";;; Module\n\n{}\n\n;; Atoms\n{}",
            self.variables_to_smtlib(),
            self.0
                .atoms()
                .iter()
                .enumerate()
                .map(|(n, atom)| format!(
                    "\n;; --- Atom {} ---\n{}",
                    n,
                    AtomSmtLibTranslator(atom).methods_to_smtlib()
                ))
                .collect::<Vec<String>>()
                .join("\n")
        )
    }

    pub fn init_to_smtlib(&self) -> String {
        self.0
            .atoms()
            .iter()
            .enumerate()
            .map(|(n, atom)| {
                format!(
                    "\n;; --- Atom {} ---\n{}",
                    n,
                    AtomSmtLibTranslator(atom).init().join("\n")
                )
            })
            .collect::<Vec<String>>()
            .join("\n")
    }

    pub fn update_to_smtlib(&self) -> String {
        self.0
            .atoms()
            .iter()
            .enumerate()
            .map(|(n, atom)| {
                format!(
                    "\n;; --- Atom {} ---\n{}",
                    n,
                    AtomSmtLibTranslator(atom).update().join("\n")
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
