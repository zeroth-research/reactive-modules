use std::collections::HashSet;
use std::fmt;

use crate::dtype::DType;
use crate::itype::{ArithOp, CmpOp, IType, LogicalOp, Val};

use base::{Atom, Module, Term};

type SmtAtom = Atom<DType, IType>;
type SmtModule = Module<DType, IType>;

pub struct AtomSmtLibTranslator<'a>(&'a SmtAtom);

pub fn wires_decls<'a>(
    wires: impl IntoIterator<Item = &'a base::Wire<DType>>,
    w: &mut impl fmt::Write,
) -> fmt::Result {
    for wire in wires {
        writeln!(w, "{}", declare_var(wire.id(), wire.dtype()))?;
    }
    Ok(())
}

impl AtomSmtLibTranslator<'_> {
    fn method_to_smtlib<'a>(
        &self,
        terms: impl IntoIterator<Item = &'a Term<DType, IType>>,
        state_wires: &HashSet<usize>,
        w: &mut impl fmt::Write,
    ) -> fmt::Result {
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
            for s in &state_assigns {
                writeln!(w, "(assert {})", s)?;
            }
        } else {
            writeln!(
                w,
                "(assert\n  (let ({})\n    (and {})))",
                let_bindings.join("\n        "),
                state_assigns.join("\n         ")
            )?;
        }
        Ok(())
    }

    pub fn init(&self, state_wires: &HashSet<usize>, w: &mut impl fmt::Write) -> fmt::Result {
        self.method_to_smtlib(self.0.init().iter(), state_wires, w)
    }

    pub fn update(&self, state_wires: &HashSet<usize>, w: &mut impl fmt::Write) -> fmt::Result {
        self.method_to_smtlib(self.0.update().iter(), state_wires, w)
    }

    pub fn ctrl(&self, w: &mut impl fmt::Write) -> fmt::Result {
        wires_decls(self.0.ctrl().wires(), w)
    }

    pub fn read(&self, w: &mut impl fmt::Write) -> fmt::Result {
        wires_decls(self.0.read().wires(), w)
    }

    pub fn wait(&self, w: &mut impl fmt::Write) -> fmt::Result {
        wires_decls(self.0.wait().wires(), w)
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

    /// Write smtlib declarations of `intf` variables
    pub fn intf(&self, w: &mut impl fmt::Write) -> fmt::Result {
        let vars = self.0.intf();
        for wire in vars.latched().iter().chain(vars.next().iter()) {
            writeln!(w, "{}", declare_var(wire.id(), wire.dtype()))?;
        }
        Ok(())
    }

    /// Write smtlib declarations of `extl` variables
    pub fn extl(&self, w: &mut impl fmt::Write) -> fmt::Result {
        let vars = self.0.extl();
        for wire in vars.latched().iter().chain(vars.next().iter()) {
            writeln!(w, "{}", declare_var(wire.id(), wire.dtype()))?;
        }
        Ok(())
    }

    pub fn variables_to_smtlib(&self, w: &mut impl fmt::Write) -> fmt::Result {
        writeln!(w, ";; Interface")?;
        self.intf(w)?;
        writeln!(w, "\n;; External")?;
        self.extl(w)
    }

    pub fn to_smtlib(&self, w: &mut impl fmt::Write) -> fmt::Result {
        if !self.0.prvt().is_empty() {
            unimplemented!()
        }

        let state_wires = self.state_wire_ids();

        writeln!(w, ";;; Module\n")?;
        self.variables_to_smtlib(w)?;
        writeln!(w, "\n;; Atoms")?;

        for (n, atom) in self.0.atoms().iter().enumerate() {
            let translator = AtomSmtLibTranslator(atom);
            writeln!(w, "\n;; --- Atom {} ---", n)?;
            writeln!(w, ";; Init")?;
            translator.init(&state_wires, w)?;
            writeln!(w, "\n;; Update")?;
            translator.update(&state_wires, w)?;
        }
        Ok(())
    }

    pub fn init_to_smtlib(&self, w: &mut impl fmt::Write) -> fmt::Result {
        let state_wires = self.state_wire_ids();
        for (n, atom) in self.0.atoms().iter().enumerate() {
            writeln!(w, "\n;; --- Atom {} ---", n)?;
            AtomSmtLibTranslator(atom).init(&state_wires, w)?;
        }
        Ok(())
    }

    pub fn update_to_smtlib(&self, w: &mut impl fmt::Write) -> fmt::Result {
        let state_wires = self.state_wire_ids();
        for (n, atom) in self.0.atoms().iter().enumerate() {
            writeln!(w, "\n;; --- Atom {} ---", n)?;
            AtomSmtLibTranslator(atom).update(&state_wires, w)?;
        }
        Ok(())
    }
}

pub fn module_to_smtlib(module: &Module<DType, IType>, w: &mut impl fmt::Write) -> fmt::Result {
    ModuleSmtLibTranslator(module).to_smtlib(w)
}

pub fn module_init_to_smtlib(
    module: &Module<DType, IType>,
    w: &mut impl fmt::Write,
) -> fmt::Result {
    ModuleSmtLibTranslator(module).init_to_smtlib(w)
}

pub fn module_update_to_smtlib(
    module: &Module<DType, IType>,
    w: &mut impl fmt::Write,
) -> fmt::Result {
    ModuleSmtLibTranslator(module).update_to_smtlib(w)
}

pub fn module_variables_to_smtlib(
    module: &Module<DType, IType>,
    w: &mut impl fmt::Write,
) -> fmt::Result {
    ModuleSmtLibTranslator(module).variables_to_smtlib(w)
}

pub fn parse_modules(
    modules: &[Module<DType, IType>],
    w: &mut impl fmt::Write,
) -> fmt::Result {
    for module in modules {
        module_to_smtlib(module, w)?;
    }
    Ok(())
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
