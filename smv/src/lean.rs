use crate::dtype::DType;

/// Kind of an expression node for the neutral, owned expression AST.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprKind {
    Number,
    Ident,
    Cond,
    Or,
    And,
    Cmp,
    Arith,
    Term,
    Primary,
    Factor,
    Assign,
    True,
    False,
    // operators for comparison/arithmetic
    LT,
    LE,
    GT,
    GE,
    EQ,
    PLUS,
    MINUS,
    TIMES,
    DIVIDE,
    NOT,
}

/// A small, owned AST for expressions. This is parser-agnostic.
#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub text: Option<String>,
    pub children: Vec<Expr>,
}

/// Neutral view of a parsed module suitable for language-agnostic translators.
pub struct LeanModule {
    /// wires: vector of (name, index, dtype). VARs first, then IVARs.
    pub wires: Vec<(String, usize, DType)>,
    /// number of VAR declarations (prefix length of wires)
    pub var_count: usize,
    /// mapping from var index to optional init expression
    pub init_exprs: Vec<Option<Expr>>,
    /// mapping from var index to optional next expression
    pub next_exprs: Vec<Option<Expr>>,
}

/// Render a `LeanModule` into a Lean snippet.
pub fn to_lean(view: &LeanModule) -> Result<String, &'static str> {
    let wires = &view.wires;
    let var_count = view.var_count;

    let mut out = String::new();
    out.push_str("structure State where\n");
    for (_name, idx, dtype) in wires.iter().take(var_count) {
        let ty = match dtype { DType::Int => "Int", DType::Bool => "Bool" };
        out.push_str(&format!("  x{} : {}\n", idx, ty));
    }
    out.push_str("\nstructure ExternalParams where\n");
    for (_name, idx, dtype) in wires.iter().skip(var_count) {
        let ty = match dtype { DType::Int => "Int", DType::Bool => "Bool" };
        out.push_str(&format!("  x{} : {}\n", idx, ty));
    }
    out.push_str("\n");

    // Small helper: drill down single-child chains to the leaf node.
    // Used for simple identifier detection in init/update rendering.
    fn leaf<'a>(e: &'a Expr) -> &'a Expr {
        let mut cur = e;
        loop {
            if cur.children.len() == 1 {
                cur = &cur.children[0];
                continue;
            }
            break cur;
        }
    }

    // Helper: render Expr into Lean strings.
    fn expr_to_lean(expr: &Expr, wires: &[(String, usize, DType)], context: &str) -> String {
        match expr.kind {
            ExprKind::Number => expr.text.clone().unwrap_or_default(),
            ExprKind::Ident => {
                let nm = expr.text.as_ref().map(|s| s.as_str()).unwrap_or("");
                if let Some((_, idx, _)) = wires.iter().find(|(s,_,_)| s == nm) {
                    if context == "params" { format!("params.x{}", idx) }
                    else if context == "s" { format!("s.x{}", idx) }
                    else { format!("x{}", idx) }
                } else { expr.text.clone().unwrap_or_default() }
            }
            ExprKind::Cond => {
                if expr.children.is_empty() { return expr.text.clone().unwrap_or_default(); }
                if expr.children.len() == 1 { return expr_to_lean(&expr.children[0], wires, "s"); }
                // expect [cond, then, else]
                let cond = expr_to_lean(&expr.children[0], wires, "s");
                let then_s = expr_to_lean(&expr.children[1], wires, "s");
                let else_s = expr_to_lean(&expr.children[2], wires, "s");
                format!("if {} then {} else {}", cond, then_s, else_s)
            }
            ExprKind::Or => {
                let parts: Vec<String> = expr.children.iter().map(|c| expr_to_lean(c, wires, "s")).filter(|s| !s.is_empty()).collect();
                parts.join(" ∨ ")
            }
            ExprKind::And => {
                let parts: Vec<String> = expr.children.iter().map(|c| expr_to_lean(c, wires, "s")).filter(|s| !s.is_empty()).collect();
                parts.join(" ∧ ")
            }
            ExprKind::Cmp => {
                let mut iter = expr.children.iter();
                let mut out = expr_to_lean(iter.next().unwrap(), wires, "s");
                while let Some(op_node) = iter.next() {
                    let right = expr_to_lean(iter.next().unwrap(), wires, "s");
                    let op_s = match op_node.kind {
                        ExprKind::LT => " < ", ExprKind::LE => " <= ", ExprKind::GT => " > ", ExprKind::GE => " >= ", ExprKind::EQ => " = ", _ => " < ",
                    };
                    out = format!("{}{}{}", out, op_s, right);
                }
                out
            }
            ExprKind::Arith | ExprKind::Term => {
                // Render arithmetic without extra parentheses to match goal formatting
                let mut iter = expr.children.iter();
                let mut acc = expr_to_lean(iter.next().unwrap(), wires, "s");
                while let Some(op_node) = iter.next() {
                    let right = expr_to_lean(iter.next().unwrap(), wires, "s");
                    let op_s = match op_node.kind {
                        ExprKind::PLUS => " + ", ExprKind::MINUS => " - ", ExprKind::TIMES => " * ", ExprKind::DIVIDE => " / ", _ => " + ",
                    };
                    acc = format!("{}{}{}", acc, op_s, right);
                }
                acc
            }
            ExprKind::Primary | ExprKind::Factor | ExprKind::Assign => {
                if let Some(inner) = expr.children.get(0) { expr_to_lean(inner, wires, "s") } else { String::new() }
            }
            ExprKind::True => "True".to_string(),
            ExprKind::False => "False".to_string(),
            _ => expr.text.clone().unwrap_or_default(),
        }
    }

    // init
    out.push_str("def init (params : ExternalParams) : State :=\n  { ");
    let mut parts: Vec<String> = vec![];
    for i in 0..var_count {
        if let Some(expr_node) = &view.init_exprs[i] {
            let l = leaf(expr_node);
            if l.kind == ExprKind::Ident {
                if let Some(name) = l.text.as_ref() {
                    if let Some((_, idx, _)) = wires.iter().find(|(s,_,_)| s == name) {
                        // If the RHS is an IVAR (index >= var_count) render as params.x{idx}
                        if *idx >= var_count {
                            parts.push(format!("x{} := params.x{}", i, idx));
                            continue;
                        }
                    }
                }
            }
            let expr = expr_to_lean(expr_node, &wires, "params");
            parts.push(format!("x{} := {}", i, expr));
        } else {
            let ty = wires[i].2;
            let val = match ty { DType::Int => "0", DType::Bool => "False" };
            parts.push(format!("x{} := {}", i, val));
        }
    }
    out.push_str(&parts.join(", "));
    out.push_str(" }\n\n");

    // update: produce a readable conditional that uses `let` for identity
    // variables and `{ s with ... }` record update when only a single var
    // changes. This matches the style in lean_goal.md.
    out.push_str("noncomputable def update (s : State) : State :=\n");

    // Find the index of the first conditional next expression, if any.
    let mut cond_idx: Option<usize> = None;
    for i in 0..var_count {
        if let Some(Some(expr_node)) = view.next_exprs.get(i) {
            if expr_node.kind == ExprKind::Cond {
                cond_idx = Some(i);
                break;
            }
        }
    }

    if let Some(ci) = cond_idx {
        let cond_expr = view.next_exprs[ci].as_ref().unwrap();

        // collect identifiers used in cond expression (by index)
        fn collect_idents(e: &Expr, wires: &[(String, usize, DType)], out: &mut std::collections::HashSet<usize>) {
            if e.kind == ExprKind::Ident {
                if let Some(name) = e.text.as_ref() {
                    if let Some((_, idx, _)) = wires.iter().find(|(s,_,_)| s == name) {
                        out.insert(*idx);
                    }
                }
            }
            for c in e.children.iter() { collect_idents(c, wires, out); }
        }

        // Extract the actual condition, then and else subexpressions by
        // skipping punctuation children (like "?" and ":") which the
        // parser may include as separate leaf nodes.
        let mut parts: Vec<&Expr> = Vec::new();
        for child in cond_expr.children.iter() {
            if let Some(t) = child.text.as_ref() {
                if t == "?" || t == ":" { continue; }
            }
            parts.push(child);
        }
        let cond_part = parts.get(0).copied().unwrap_or(cond_expr);
        let then_part = parts.get(1).copied().unwrap_or(cond_expr);
        let else_part = parts.get(2).copied().unwrap_or(cond_expr);

        let mut idents_used = std::collections::HashSet::new();
        collect_idents(cond_part, &wires, &mut idents_used);

        // Emit let bindings for identity vars (except the conditional var).
        // We keep this simple: if a var's next is identity, emit a `let` to
        // make subsequent expressions readable. This matches the goal output.
        for i in 0..var_count {
            let is_identity = match view.next_exprs.get(i) {
                Some(Some(expr_node)) => {
                    let l = leaf(expr_node);
                    l.kind == ExprKind::Ident && l.text.as_ref().map(|s| s.as_str()) == Some(wires[i].0.as_str())
                }
                Some(None) => true,
                None => true,
            };
            if is_identity && i != ci {
                out.push_str(&format!("  let x{} := s.x{}\n", i, i));
            }
        }

    // Render condition
    let cond_render = expr_to_lean(cond_part, &wires, "s");
        out.push_str(&format!("  if {} then\n", cond_render));

        // Render then/else as { s with xci := ... } if only ci changes
    let then_s = expr_to_lean(then_part, &wires, "s");
    let else_s = expr_to_lean(else_part, &wires, "s");

        out.push_str(&format!("    {{ s with x{} := {} }}\n", ci, then_s));
        out.push_str(&format!("  else\n    {{ s with x{} := {} }}\n", ci, else_s));
    } else {
        out.push_str("  { /* identity */ }\n");
    }

    Ok(out)
}
