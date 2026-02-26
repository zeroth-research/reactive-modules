use crate::{DType, IType};

use super::tensor::tensor_to_lean;

pub struct AtomTranslator<'a>(&'a base::Atom<DType, IType>);

impl AtomTranslator<'_> {
    fn translate_terms<'a, T>(&self, terms: T) -> Vec<String>
    where
        T: IntoIterator<Item = &'a base::Term<DType, IType>>,
    {
        let mut res: Vec<String> = Vec::new();
        for term in terms {
            let wire_id = term.write().ids().next().unwrap();
            let expr = to_expr(term);
            res.push(format!("let {} := {}", wire_name(wire_id), expr));
        }

        res
        //
        // if let_bindings.is_empty() {
        //     state_assigns
        //         .iter()
        //         .map(|s| format!("(assert {})", s))
        //         .collect()
        // } else {
        //     vec![format!(
        //         "(assert\n  (let ({})\n    (and {})))",
        //         let_bindings.join("\n        "),
        //         state_assigns.join("\n         ")
        //     )]
        // }
    }

    pub fn init(&self) -> Vec<String> {
        self.translate_terms(self.0.init().iter())
    }

    pub fn update(&self) -> Vec<String> {
        self.translate_terms(self.0.update().iter())
    }
}

pub struct ModuleTranslator<'a>(pub(crate) &'a base::Module<DType, IType>);

fn get_params(wires: &base::Interface<DType, 2>, idx: usize) -> String {
    let params: Vec<String> = if idx == 0 {
        wires
            .iter()
            .map(|[wl, _]| wire_name(wl.id()))
            .collect::<Vec<String>>()
    } else if idx == 1 {
        wires
            .iter()
            .map(|[_, wu]| wire_name(wu.id()))
            .collect::<Vec<String>>()
    } else {
        panic!("BUG: invalid index")
    };

    if params.len() > 1 {
        format!("({})", params.join(", "))
    } else {
        params.join(" ")
    }
}

impl ModuleTranslator<'_> {
    pub fn to_lean(&self) -> String {
        format!("{}\n\n{}", self.translate_init(), self.translate_update())
    }
    pub fn translate_init(&self) -> String {
        let params = get_params(self.0.extl(), 1);
        let retval = get_params(self.0.intf(), 1);

        format!(
            "def init {} :=\n{}\n  {}",
            params,
            self.0
                .atoms()
                .iter()
                .map(|atom| { format!("  {}", AtomTranslator(atom).init().join("\n  ")) })
                .collect::<Vec<String>>()
                .join("\n"),
            retval
        )
    }

    pub fn translate_update(&self) -> String {
        let intf = get_params(self.0.intf(), 0);
        let extl = get_params(self.0.extl(), 0);
        let extl_n = get_params(self.0.extl(), 1);
        let retval = get_params(self.0.intf(), 1);

        format!(
            "def update {} {} {} :=\n{}\n  {}",
            intf,
            extl,
            extl_n,
            self.0
                .atoms()
                .iter()
                .map(|atom| { format!("  {}", AtomTranslator(atom).update().join("\n  ")) })
                .collect::<Vec<String>>()
                .join("\n"),
            retval
        )
    }
}

fn wire_name(id: usize) -> String {
    format!("w{}", id)
}

fn to_expr(term: &base::Term<DType, IType>) -> String {
    let args: Vec<String> = term.read().wires().map(|w| wire_name(w.id())).collect();
    match term.itype() {
        // arithmetic
        IType::Add() => format!("({} + {})", args[0], args[1]),
        IType::Sub() => format!("({} - {})", args[0], args[1]),
        IType::Mul() => format!("({} * {})", args[0], args[1]),
        IType::Div() => format!("({} / {})", args[0], args[1]),
        // comparisons
        IType::Eq() => format!("({} = {})", args[0], args[1]),
        IType::Le() => format!("({} ≤ {})", args[0], args[1]),
        IType::Ge() => format!("({} ≥ {})", args[0], args[1]),
        IType::Lt() => format!("({} < {})", args[0], args[1]),
        IType::Gt() => format!("({} > {})", args[0], args[1]),
        // boolean operations
        IType::And() => format!("({} ∧ {})", args[0], args[1]),
        IType::Or() => format!("({} ∨ {})", args[0], args[1]),
        IType::Not() => format!("(¬ {})", args[0]),
        // tensors
        IType::Tensor(t) => {
            debug_assert!(args.is_empty());
            tensor_to_lean(&t.tensor)
        }
        IType::Argmax() => format!("(argmax {})", args[0]),
        // other
        IType::Id() => format!("{}", args[0]),
        IType::Ite() => format!("(if {} then {} else {})", args[0], args[1], args[2]),
        _ => format!("TODO {}", term),
    }
}
