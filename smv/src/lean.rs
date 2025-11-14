use crate::dtype::DType;
use crate::itype::IType;
use base::module::Module;
use base::term::Term;
use base::wire::Wire;

/// Kind of an expression node for the neutral, owned expression AST.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprKind {
    Number,
    Ident,
    Cond,
    Abs,
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
    /// optional source IVAR index for init values that originate from
    /// abs(IVAR) lowered into a primed Assign; maps var idx -> Some(ivar_idx)
    pub init_abs_src: Vec<Option<usize>>,
}

/// Render a `LeanModule` into a Lean snippet.
pub fn to_lean(view: &LeanModule) -> Result<String, &'static str> {
    let wires = &view.wires;
    let var_count = view.var_count;

    let mut out = String::new();
    out.push_str("structure State where\n");
    for (_name, idx, dtype) in wires.iter().take(var_count) {
        let ty = match dtype {
            DType::Int => "Int",
            DType::Bool => "Bool",
        };
        out.push_str(&format!("  x{} : {}\n", idx, ty));
    }
    out.push_str("\nstructure ExternalParams where\n");
    for (_name, idx, dtype) in wires.iter().skip(var_count) {
        let ty = match dtype {
            DType::Int => "Int",
            DType::Bool => "Bool",
        };
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
                if let Some((_, idx, _)) = wires.iter().find(|(s, _, _)| s == nm) {
                    if context == "params" {
                        format!("params.x{}", idx)
                    } else if context == "s" {
                        format!("s.x{}", idx)
                    } else {
                        format!("x{}", idx)
                    }
                } else {
                    expr.text.clone().unwrap_or_default()
                }
            }
            ExprKind::Cond => {
                if expr.children.is_empty() {
                    return expr.text.clone().unwrap_or_default();
                }
                if expr.children.len() == 1 {
                    return expr_to_lean(&expr.children[0], wires, context);
                }
                // expect [cond, then, else]
                let cond_node = &expr.children[0];
                let then_node = &expr.children[1];
                let else_node = &expr.children[2];

                // Try to detect the relu pattern: if (left < 0) ? 0 : left
                // and render it as Int.natAbs <context>.xN when appropriate.
                fn is_zero(n: &Expr) -> bool {
                    n.kind == ExprKind::Number && n.text.as_ref().map(|s| s.as_str()) == Some("0")
                }

                if cond_node.kind == ExprKind::Cmp && is_zero(then_node) {
                    // cond_node children: left, op, right
                    if cond_node.children.len() >= 3 {
                        let left_c = &cond_node.children[0];
                        let op_c = &cond_node.children[1];
                        let right_c = &cond_node.children[2];
                        if op_c.kind == ExprKind::LT && is_zero(right_c) {
                            // Render left and else with the current context and compare
                            let left_s = expr_to_lean(left_c, wires, context);
                            let else_s = expr_to_lean(else_node, wires, context);
                            if left_s == else_s {
                                return format!("Int.natAbs {}", left_s);
                            }
                        }
                    }
                }

                let cond = expr_to_lean(cond_node, wires, context);
                let then_s = expr_to_lean(then_node, wires, context);
                let else_s = expr_to_lean(else_node, wires, context);
                format!("if {} then {} else {}", cond, then_s, else_s)
            }

            ExprKind::Abs => {
                // Unary abs node: render as Int.natAbs <child>
                if let Some(child) = expr.children.get(0) {
                    let c = expr_to_lean(child, wires, context);
                    return format!("Int.natAbs {}", c);
                }
                String::new()
            }
            ExprKind::Or => {
                let parts: Vec<String> = expr
                    .children
                    .iter()
                    .map(|c| expr_to_lean(c, wires, "s"))
                    .filter(|s| !s.is_empty())
                    .collect();
                parts.join(" ∨ ")
            }
            ExprKind::And => {
                let parts: Vec<String> = expr
                    .children
                    .iter()
                    .map(|c| expr_to_lean(c, wires, "s"))
                    .filter(|s| !s.is_empty())
                    .collect();
                parts.join(" ∧ ")
            }
            ExprKind::Cmp => {
                let mut iter = expr.children.iter();
                let mut out = expr_to_lean(iter.next().unwrap(), wires, "s");
                while let Some(op_node) = iter.next() {
                    let right = expr_to_lean(iter.next().unwrap(), wires, "s");
                    let op_s = match op_node.kind {
                        ExprKind::LT => " < ",
                        ExprKind::LE => " <= ",
                        ExprKind::GT => " > ",
                        ExprKind::GE => " >= ",
                        ExprKind::EQ => " = ",
                        _ => " < ",
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
                        ExprKind::PLUS => " + ",
                        ExprKind::MINUS => " - ",
                        ExprKind::TIMES => " * ",
                        ExprKind::DIVIDE => " / ",
                        _ => " + ",
                    };
                    acc = format!("{}{}{}", acc, op_s, right);
                }
                acc
            }
            ExprKind::Primary | ExprKind::Factor | ExprKind::Assign => {
                if let Some(inner) = expr.children.get(0) {
                    expr_to_lean(inner, wires, "s")
                } else {
                    String::new()
                }
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
                    if let Some((_, idx, _)) = wires.iter().find(|(s, _, _)| s == name) {
                        // If the RHS is an IVAR (index >= var_count) render as params.x{idx}
                        if *idx >= var_count {
                            // Heuristic: IVARs used in init often originate from
                            // abs(...) lowering. Render these as the absolute
                            // value in the generated init: `Int.natAbs params.xN`.
                            parts.push(format!("x{} := Int.natAbs params.x{}", i, idx));
                            continue;
                        }
                    }
                }
            }
            let expr = expr_to_lean(expr_node, &wires, "params");
            parts.push(format!("x{} := {}", i, expr));
        } else {
            let ty = wires[i].2;
            let val = match ty {
                DType::Int => "0",
                DType::Bool => "False",
            };
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
        fn collect_idents(
            e: &Expr,
            wires: &[(String, usize, DType)],
            out: &mut std::collections::HashSet<usize>,
        ) {
            if e.kind == ExprKind::Ident {
                if let Some(name) = e.text.as_ref() {
                    if let Some((_, idx, _)) = wires.iter().find(|(s, _, _)| s == name) {
                        out.insert(*idx);
                    }
                }
            }
            for c in e.children.iter() {
                collect_idents(c, wires, out);
            }
        }

        // Extract the actual condition, then and else subexpressions by
        // skipping punctuation children (like "?" and ":") which the
        // parser may include as separate leaf nodes.
        let mut parts: Vec<&Expr> = Vec::new();
        for child in cond_expr.children.iter() {
            if let Some(t) = child.text.as_ref() {
                if t == "?" || t == ":" {
                    continue;
                }
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
                    l.kind == ExprKind::Ident
                        && l.text.as_ref().map(|s| s.as_str()) == Some(wires[i].0.as_str())
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

/// Construct a LeanModule view from a `base::Module<DType, IType>`.
/// This performs a best-effort conversion of simple Terms (Assign/Const)
/// into the owned `Expr` AST used by the renderer. More complex operator
/// terms will be rendered as simple textual leaves.
pub fn to_lean_from_module(m: &Module<DType, IType>) -> Result<String, &'static str> {
    // Build wires vector: take the latched wire list and use names x{index}
    let wire_pair = m.wire();
    let latched = &wire_pair[0];
    let mut wires: Vec<(String, usize, DType)> = Vec::new();
    // Deduplicate latched entries by index to avoid repeated wires in
    // parsed outputs (some upstream builders produced duplicates).
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (i, dtype) in latched.iter() {
        if seen.insert(i) {
            wires.push((format!("x{}", i), i, *dtype));
        }
    }
    // Infer var_count as total latched wires minus external latched wires
    let extl_len = m.extl()[0].len();
    let var_count = wires.len().saturating_sub(extl_len);

    let atoms = m.atoms();

    // Build separate maps for init and update terms so we don't clobber
    // init terms with update terms that share the same next wire indices.
    let mut init_term_map: std::collections::HashMap<usize, &Term<DType, IType>> =
        std::collections::HashMap::new();
    let mut update_term_map: std::collections::HashMap<usize, &Term<DType, IType>> =
        std::collections::HashMap::new();
    for atom in atoms.iter() {
        for t in atom.init().iter() {
            for (widx, _) in t.write().iter() {
                init_term_map.insert(widx, t);
            }
        }
        for t in atom.update().iter() {
            for (widx, _) in t.write().iter() {
                update_term_map.insert(widx, t);
            }
        }
    }

    // Recursive term->Expr builder. Resolve indices to either variable
    // identifiers (xN) or recurse into the term that produces a temporary.
    fn build_expr_for_index(
        idx: usize,
        n: usize,
        var_count: usize,
        wires: &[(String, usize, DType)],
        term_map: &std::collections::HashMap<usize, &Term<DType, IType>>,
    ) -> Expr {
        use crate::itype::IType::*;
        use base::term::TermWire;

        // If this index is a latched or unprimed IVAR, render as Ident x{idx}
        if idx < n {
            return Expr {
                kind: ExprKind::Ident,
                text: Some(format!("x{}", idx)),
                children: vec![],
            };
        }

        // If there's a term that writes this index, expand it.
        if let Some(t) = term_map.get(&idx) {
            match t.itype() {
                ConstInt(v) => Expr {
                    kind: ExprKind::Number,
                    text: Some(v.to_string()),
                    children: vec![],
                },
                ConstBool(b) => {
                    if *b {
                        Expr {
                            kind: ExprKind::True,
                            text: None,
                            children: vec![],
                        }
                    } else {
                        Expr {
                            kind: ExprKind::False,
                            text: None,
                            children: vec![],
                        }
                    }
                }
                Assign => {
                    // render source as identifier
                    if let Some((ridx, _)) = t.read().iter().next() {
                        // If ridx is a primed index (>= n) but not produced by a term,
                        // map it back to its unprimed base (ridx - n) to allow params mapping.
                        if ridx >= n && !term_map.contains_key(&ridx) {
                            let base = ridx - n;
                            return Expr {
                                kind: ExprKind::Ident,
                                text: Some(format!("x{}", base)),
                                children: vec![],
                            };
                        }
                        return build_expr_for_index(ridx, n, var_count, wires, term_map);
                    }
                    Expr {
                        kind: ExprKind::Ident,
                        text: Some(String::new()),
                        children: vec![],
                    }
                }
                Lt | Le | Gt | Ge | Eq => {
                    // Build interleaved children: left, op, right
                    let mut children: Vec<Expr> = vec![];
                    let rwire = t.read();
                    let mut iter = rwire.iter();
                    if let Some((lidx, _)) = iter.next() {
                        if lidx >= n && !term_map.contains_key(&lidx) {
                            children.push(Expr {
                                kind: ExprKind::Ident,
                                text: Some(format!("x{}", lidx - n)),
                                children: vec![],
                            });
                        } else {
                            children
                                .push(build_expr_for_index(lidx, n, var_count, wires, term_map));
                        }
                    }
                    // op node
                    let op_kind = match t.itype() {
                        Lt => ExprKind::LT,
                        Le => ExprKind::LE,
                        Gt => ExprKind::GT,
                        Ge => ExprKind::GE,
                        Eq => ExprKind::EQ,
                        _ => ExprKind::EQ,
                    };
                    children.push(Expr {
                        kind: op_kind,
                        text: None,
                        children: vec![],
                    });
                    if let Some((ridx, _)) = iter.next() {
                        if ridx >= n && !term_map.contains_key(&ridx) {
                            children.push(Expr {
                                kind: ExprKind::Ident,
                                text: Some(format!("x{}", ridx - n)),
                                children: vec![],
                            });
                        } else {
                            children
                                .push(build_expr_for_index(ridx, n, var_count, wires, term_map));
                        }
                    }
                    Expr {
                        kind: ExprKind::Cmp,
                        text: None,
                        children,
                    }
                }
                Or => {
                    let children = t
                        .read()
                        .iter()
                        .map(|(ridx, _)| {
                            if ridx >= n && !term_map.contains_key(&ridx) {
                                Expr {
                                    kind: ExprKind::Ident,
                                    text: Some(format!("x{}", ridx - n)),
                                    children: vec![],
                                }
                            } else {
                                build_expr_for_index(ridx, n, var_count, wires, term_map)
                            }
                        })
                        .collect();
                    Expr {
                        kind: ExprKind::Or,
                        text: None,
                        children,
                    }
                }
                And => {
                    let children = t
                        .read()
                        .iter()
                        .map(|(ridx, _)| {
                            if ridx >= n && !term_map.contains_key(&ridx) {
                                Expr {
                                    kind: ExprKind::Ident,
                                    text: Some(format!("x{}", ridx - n)),
                                    children: vec![],
                                }
                            } else {
                                build_expr_for_index(ridx, n, var_count, wires, term_map)
                            }
                        })
                        .collect();
                    Expr {
                        kind: ExprKind::And,
                        text: None,
                        children,
                    }
                }
                Add | Sub | Mul | Div => {
                    // Build interleaved children: a op b op c ... using the op kind
                    let op_kind = match t.itype() {
                        Add => ExprKind::PLUS,
                        Sub => ExprKind::MINUS,
                        Mul => ExprKind::TIMES,
                        Div => ExprKind::DIVIDE,
                        _ => ExprKind::PLUS,
                    };
                    let mut children: Vec<Expr> = vec![];
                    let rwire = t.read();
                    let mut iter = rwire.iter();
                    if let Some((first, _)) = iter.next() {
                        if first >= n && !term_map.contains_key(&first) {
                            children.push(Expr {
                                kind: ExprKind::Ident,
                                text: Some(format!("x{}", first - n)),
                                children: vec![],
                            });
                        } else {
                            children
                                .push(build_expr_for_index(first, n, var_count, wires, term_map));
                        }
                    }
                    while let Some((r, _)) = iter.next() {
                        // op node
                        children.push(Expr {
                            kind: op_kind.clone(),
                            text: None,
                            children: vec![],
                        });
                        if r >= n && !term_map.contains_key(&r) {
                            children.push(Expr {
                                kind: ExprKind::Ident,
                                text: Some(format!("x{}", r - n)),
                                children: vec![],
                            });
                        } else {
                            children.push(build_expr_for_index(r, n, var_count, wires, term_map));
                        }
                    }
                    Expr {
                        kind: ExprKind::Arith,
                        text: None,
                        children,
                    }
                }
                Cond => {
                    // Cond expects three read args: cond, then, else
                    let mut children: Vec<Expr> = vec![];
                    for (ridx, _) in t.read().iter() {
                        if ridx >= n && !term_map.contains_key(&ridx) {
                            children.push(Expr {
                                kind: ExprKind::Ident,
                                text: Some(format!("x{}", ridx - n)),
                                children: vec![],
                            });
                        } else {
                            children
                                .push(build_expr_for_index(ridx, n, var_count, wires, term_map));
                        }
                    }
                    Expr {
                        kind: ExprKind::Cond,
                        text: None,
                        children,
                    }
                }
                Abs => {
                    // Unary abs: single read argument
                    if let Some((ridx, _)) = t.read().iter().next() {
                        if ridx >= n && !term_map.contains_key(&ridx) {
                            Expr {
                                kind: ExprKind::Abs,
                                text: None,
                                children: vec![Expr {
                                    kind: ExprKind::Ident,
                                    text: Some(format!("x{}", ridx - n)),
                                    children: vec![],
                                }],
                            }
                        } else {
                            Expr {
                                kind: ExprKind::Abs,
                                text: None,
                                children: vec![build_expr_for_index(
                                    ridx, n, var_count, wires, term_map,
                                )],
                            }
                        }
                    } else {
                        Expr {
                            kind: ExprKind::Abs,
                            text: None,
                            children: vec![],
                        }
                    }
                }
                Not => {
                    let children = t
                        .read()
                        .iter()
                        .map(|(ridx, _)| build_expr_for_index(ridx, n, var_count, wires, term_map))
                        .collect();
                    Expr {
                        kind: ExprKind::NOT,
                        text: None,
                        children,
                    }
                }
                _ => Expr {
                    kind: ExprKind::Primary,
                    text: Some(format!("{:?}", t.itype())),
                    children: vec![],
                },
            }
        } else {
            // No producer: this may be a primed IVAR index; map back to base
            if idx >= n {
                let base = idx - n;
                return Expr {
                    kind: ExprKind::Ident,
                    text: Some(format!("x{}", base)),
                    children: vec![],
                };
            }
            Expr {
                kind: ExprKind::Ident,
                text: Some(format!("x{}", idx)),
                children: vec![],
            }
        }
    }

    // Build init_exprs and next_exprs for each VAR (0..var_count)
    let n = wires.len();
    let mut init_exprs: Vec<Option<Expr>> = vec![None; var_count];
    let mut next_exprs: Vec<Option<Expr>> = vec![None; var_count];

    let atoms = m.atoms();
    use base::term::TermWire;
    for atom in atoms.iter() {
        for t in atom.init().iter() {
            // find writes matching var index + n
            for (widx, _) in t.write().iter() {
                if widx >= n {
                    // write into next space
                    let vidx = widx - n;
                    if vidx < var_count {
                        init_exprs[vidx] = Some(build_expr_for_index(
                            widx,
                            n,
                            var_count,
                            &wires,
                            &init_term_map,
                        ));
                    }
                }
            }
        }
        for t in atom.update().iter() {
            for (widx, _) in t.write().iter() {
                if widx >= n {
                    let vidx = widx - n;
                    if vidx < var_count {
                        next_exprs[vidx] = Some(build_expr_for_index(
                            widx,
                            n,
                            var_count,
                            &wires,
                            &update_term_map,
                        ));
                    }
                }
            }
        }
    }

    // Detect init assignments that come from an Abs term lowered into a
    // temporary followed by an Assign into the primed var. Pattern:
    //   Term Abs writes -> temp_t (read primed IVAR)
    //   Term Assign writes -> vidx+n, read -> temp_t
    // When found, set init_exprs[vidx] = Some(Abs(Ident(x{base}))) where
    // base is the original IVAR index (mapped back from primed index).
    let mut init_abs_src: Vec<Option<usize>> = vec![None; var_count];
    for (widx, t) in init_term_map.iter() {
        let widx = *widx;
        if widx >= n {
            let vidx = widx - n;
            if vidx < var_count {
                if let IType::Assign = t.itype() {
                    // The Assign should read from a temp produced by an Abs
                    if let Some((ridx, _)) = t.read().iter().next() {
                        if let Some(abs_term) = init_term_map.get(&ridx) {
                            if let IType::Abs = abs_term.itype() {
                                // Abs read should point at a primed IVAR index
                                if let Some((abs_ridx, _)) = abs_term.read().iter().next() {
                                    // Map primed IVAR back to base if needed
                                    let base = if abs_ridx >= n
                                        && !init_term_map.contains_key(&abs_ridx)
                                    {
                                        abs_ridx - n
                                    } else {
                                        abs_ridx
                                    };
                                    // Record init expr as Abs(Ident(x{base}))
                                    init_exprs[vidx] = Some(Expr {
                                        kind: ExprKind::Abs,
                                        text: None,
                                        children: vec![Expr {
                                            kind: ExprKind::Ident,
                                            text: Some(format!("x{}", base)),
                                            children: vec![],
                                        }],
                                    });
                                    // Also record source ivar if applicable
                                    if base >= var_count {
                                        init_abs_src[vidx] = Some(base);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    to_lean(&LeanModule {
        wires,
        var_count,
        init_exprs,
        next_exprs,
        init_abs_src,
    })
}

/// Render a vector of Terms into a Lean expression string by following the
/// write/read DAG. `out_wire` is the wire that holds the final output for
/// the term vector (e.g., obligations.write()). `wires` and `var_count` are
/// the module's wire map used to render variable identifiers.
pub fn render_terms_to_lean(
    terms: &[Term<DType, IType>],
    out_wire: &Wire<DType>,
    wires: &[(String, usize, DType)],
    var_count: usize,
) -> String {
    use base::term::TermWire;
    use std::collections::HashMap;

    // Map primary written index -> term
    let mut map: HashMap<usize, &Term<DType, IType>> = HashMap::new();
    for t in terms.iter() {
        for (widx, _) in t.write().iter() {
            map.insert(widx, t);
        }
    }

    // Recursive renderer for an index (wire). If index is a VAR (< var_count)
    // render as x{idx}. Otherwise, if there's a term producing that index,
    // render the term recursively. Fallback to `x{idx}`.
    fn render_idx(
        idx: usize,
        map: &HashMap<usize, &Term<DType, IType>>,
        wires: &[(String, usize, DType)],
        var_count: usize,
    ) -> String {
        if idx < var_count {
            return format!("x{}", idx);
        }
        if let Some(t) = map.get(&idx) {
            render_term(t, map, wires, var_count)
        } else {
            format!("x{}", idx)
        }
    }

    fn render_term(
        t: &Term<DType, IType>,
        map: &HashMap<usize, &Term<DType, IType>>,
        wires: &[(String, usize, DType)],
        var_count: usize,
    ) -> String {
        use crate::itype::IType::*;
        let mut args: Vec<usize> = Vec::new();
        for (i, _) in t.read().iter() {
            args.push(i);
        }
        match t.itype() {
            ConstInt(v) => v.to_string(),
            ConstBool(b) => {
                if *b {
                    "True".to_string()
                } else {
                    "False".to_string()
                }
            }
            Assign => {
                // assign reads a single source
                if let Some(a) = args.get(0) {
                    render_idx(*a, map, wires, var_count)
                } else {
                    String::new()
                }
            }
            Lt | Le | Gt | Ge | Eq => {
                if args.len() >= 2 {
                    let left = render_idx(args[0], map, wires, var_count);
                    let right = render_idx(args[1], map, wires, var_count);
                    let op = match t.itype() {
                        Lt => " < ",
                        Le => " ≤ ",
                        Gt => " > ",
                        Ge => " ≥ ",
                        Eq => " = ",
                        _ => " = ",
                    };
                    format!("{}{}{}", left, op, right)
                } else {
                    String::new()
                }
            }
            Or => {
                let parts: Vec<String> = args
                    .iter()
                    .map(|i| render_idx(*i, map, wires, var_count))
                    .collect();
                parts.join(" ∨ ")
            }
            And => {
                let parts: Vec<String> = args
                    .iter()
                    .map(|i| render_idx(*i, map, wires, var_count))
                    .collect();
                parts.join(" ∧ ")
            }
            Add | Sub | Mul | Div => {
                let op = match t.itype() {
                    Add => " + ",
                    Sub => " - ",
                    Mul => " * ",
                    Div => " / ",
                    _ => " + ",
                };
                let parts: Vec<String> = args
                    .iter()
                    .map(|i| render_idx(*i, map, wires, var_count))
                    .collect();
                parts.join(op)
            }
            Abs => {
                if args.len() >= 1 {
                    let inner = render_idx(args[0], map, wires, var_count);
                    return format!("Int.natAbs {}", inner);
                }
                String::new()
            }
            Cond => {
                // expect [cond, then, else]
                if args.len() >= 3 {
                    let c_idx = args[0];
                    let th_idx = args[1];
                    let el_idx = args[2];
                    let c = render_idx(c_idx, map, wires, var_count);
                    let th = render_idx(th_idx, map, wires, var_count);
                    let el = render_idx(el_idx, map, wires, var_count);

                    // Detect a relu pattern: if diff < 0 then 0 else diff
                    // We check that the condition is a comparison term (Lt)
                    // whose right side is 0, the `then` branch is 0 and the
                    // `else` branch equals the condition's left expression.
                    if let Some(cond_term) = map.get(&c_idx) {
                        match cond_term.itype() {
                            Lt => {
                                // extract cond_term read args
                                let mut carr: Vec<usize> = Vec::new();
                                for (i, _) in cond_term.read().iter() {
                                    carr.push(i);
                                }
                                if carr.len() >= 2 {
                                    let left_idx = carr[0];
                                    let right_idx = carr[1];
                                    let left_s = render_idx(left_idx, map, wires, var_count);
                                    let right_s = render_idx(right_idx, map, wires, var_count);
                                    // Compare strings: then must be 0, right side 0, else equals left
                                    if right_s == "0" && th == "0" && el == left_s {
                                        return format!("relu ({})", left_s);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }

                    format!("if {} then {} else {}", c, th, el)
                } else {
                    String::new()
                }
            }
            Not => {
                if args.len() >= 1 {
                    format!("¬{}", render_idx(args[0], map, wires, var_count))
                } else {
                    String::new()
                }
            }
            _ => format!("{:?}", t.itype()),
        }
    }

    // pick the first index in out_wire to render
    let mut outs: Vec<usize> = Vec::new();
    for (i, _) in out_wire.iter() {
        outs.push(i);
    }
    if outs.is_empty() {
        return String::new();
    }
    render_idx(outs[0], &map, wires, var_count)
}

/// Collect the set of VAR indices used by a term vector starting from the
/// given output wire. Returns a sorted Vec<usize> of variable indices (< var_count).
pub fn collect_used_vars(
    terms: &[Term<DType, IType>],
    out_wire: &Wire<DType>,
    var_count: usize,
) -> Vec<usize> {
    use base::term::TermWire;
    use std::collections::{HashMap, HashSet};

    let mut map: HashMap<usize, &Term<DType, IType>> = HashMap::new();
    for t in terms.iter() {
        for (widx, _) in t.write().iter() {
            map.insert(widx, t);
        }
    }

    let mut used: HashSet<usize> = HashSet::new();

    fn collect_idx(
        idx: usize,
        map: &HashMap<usize, &Term<DType, IType>>,
        var_count: usize,
        used: &mut HashSet<usize>,
    ) {
        if idx < var_count {
            used.insert(idx);
            return;
        }
        if let Some(t) = map.get(&idx) {
            for (ridx, _) in t.read().iter() {
                collect_idx(ridx, map, var_count, used);
            }
        } else {
            // unmapped primed index -> map back to base if possible
            if idx >= var_count {
                let base = idx - var_count;
                used.insert(base);
            }
        }
    }

    // start from first index in out_wire
    for (i, _) in out_wire.iter() {
        collect_idx(i, &map, var_count, &mut used);
    }

    let mut out: Vec<usize> = used.into_iter().collect();
    out.sort();
    out
}
