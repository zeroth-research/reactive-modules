use pest::Parser;
use pest::iterators::Pair;

use crate::smv::SMVParser;
use crate::smv::Rule;
use crate::dtype::DType;
use crate::lean::{LeanModule, Expr, ExprKind};

// Convert a pest Pair<Rule> into the neutral, owned `Expr` defined in the
// `lean` module.
fn pair_to_expr(pair: Pair<Rule>) -> Expr {
    let rule = pair.as_rule();
    let mut children: Vec<Expr> = vec![];
    for child in pair.clone().into_inner() {
        children.push(pair_to_expr(child));
    }
    let text = if children.is_empty() { Some(pair.as_str().to_string()) } else { None };
    let kind = match rule {
        Rule::number => ExprKind::Number,
        Rule::ident => ExprKind::Ident,
        Rule::expr_cond => ExprKind::Cond,
        Rule::expr_or => ExprKind::Or,
        Rule::expr_and => ExprKind::And,
        Rule::expr_cmp => ExprKind::Cmp,
        Rule::expr_arith => ExprKind::Arith,
        Rule::expr_term => ExprKind::Term,
        Rule::expr_primary => ExprKind::Primary,
        Rule::expr_factor => ExprKind::Factor,
        Rule::expr_assign => ExprKind::Assign,
        Rule::TRUE => ExprKind::True,
        Rule::FALSE => ExprKind::False,
        Rule::LT => ExprKind::LT,
        Rule::LE => ExprKind::LE,
        Rule::GT => ExprKind::GT,
        Rule::GE => ExprKind::GE,
        Rule::EQ => ExprKind::EQ,
        Rule::PLUS => ExprKind::PLUS,
        Rule::MINUS => ExprKind::MINUS,
        Rule::TIMES => ExprKind::TIMES,
        Rule::DIVIDE => ExprKind::DIVIDE,
        Rule::NOT => ExprKind::NOT,
        _ => ExprKind::Primary,
    };
    Expr { kind, text, children }
}

/// Parse SMV text into a neutral `LeanModule` view. The function keeps all
/// pest/SMV-specific parsing inside this `smv` module and returns owned
/// `Expr` values that the `lean` renderer can consume without any parser
/// lifetime dependence.
pub fn parse_smv_to_lean_view(input: &str) -> Result<LeanModule, &'static str> {
    let parsed = SMVParser::parse(Rule::file, input).map_err(|_| "parse failed")?.next().ok_or("empty parse")?;

    let mut var_decls: Vec<(String, DType)> = vec![];
    let mut ivar_decls: Vec<(String, DType)> = vec![];
    let mut init_assigns: Vec<(String, Pair<Rule>)> = vec![];
    let mut next_assigns: Vec<(String, Pair<Rule>)> = vec![];

    for section in parsed.into_inner() {
        if section.as_rule() != Rule::module_decl { continue; }
        for inner in section.into_inner() {
            if inner.as_rule() != Rule::module_body { continue; }
            for body_item in inner.into_inner() {
                match body_item.as_rule() {
                    Rule::ivar_section => {
                        for decl in body_item.into_inner().filter(|p| p.as_rule() == Rule::ivar_decl) {
                            let mut decl_iter = decl.into_inner();
                            let name = decl_iter.next().unwrap().as_str().to_string();
                            let dtype_rule = decl_iter.next().unwrap();
                            let dtype = match dtype_rule.as_str() {
                                "boolean" => DType::Bool,
                                "integer" => DType::Int,
                                _ => DType::Bool,
                            };
                            ivar_decls.push((name, dtype));
                        }
                    }
                    Rule::var_section => {
                        for decl in body_item.into_inner().filter(|p| p.as_rule() == Rule::var_decl) {
                            let mut decl_iter = decl.into_inner();
                            let name = decl_iter.next().unwrap().as_str().to_string();
                            let dtype_rule = decl_iter.next().unwrap();
                            let dtype = match dtype_rule.as_str() {
                                "boolean" => DType::Bool,
                                "integer" => DType::Int,
                                _ => DType::Bool,
                            };
                            var_decls.push((name, dtype));
                        }
                    }
                    Rule::assign_section => {
                        for assign in body_item.into_inner().filter(|p| p.as_rule() == Rule::assign_stmt) {
                            let mut parts = assign.into_inner();
                            let target_pair = parts.next().unwrap();
                            let expr_pair = parts.next().unwrap();
                            match target_pair.as_rule() {
                                Rule::init_ref => {
                                    let var_name = target_pair.into_inner().find(|p| p.as_rule() == Rule::ident).unwrap().as_str().to_string();
                                    init_assigns.push((var_name, expr_pair));
                                }
                                Rule::next_ref => {
                                    let var_name = target_pair.into_inner().find(|p| p.as_rule() == Rule::ident).unwrap().as_str().to_string();
                                    next_assigns.push((var_name, expr_pair));
                                }
                                Rule::ident => {
                                    let var_name = target_pair.as_str().to_string();
                                    init_assigns.push((var_name, expr_pair));
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Build wires ordering: VARs then IVARs. Index is their position.
    let mut wires: Vec<(String, usize, DType)> = vec![];
    for (name,dtype) in var_decls.iter() { wires.push((name.clone(), wires.len(), *dtype)); }
    for (name,dtype) in ivar_decls.iter() { wires.push((name.clone(), wires.len(), *dtype)); }
    let var_count = var_decls.len();

    // Build maps from name to index
    let mut name_to_idx: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (name, idx, _) in wires.iter() { name_to_idx.insert(name.clone(), *idx); }

    let mut init_exprs: Vec<Option<Expr>> = vec![None; var_count];
    let mut next_exprs: Vec<Option<Expr>> = vec![None; var_count];
    for (name, expr) in init_assigns.into_iter() {
        if let Some(idx) = name_to_idx.get(&name) {
            if *idx < var_count { init_exprs[*idx] = Some(pair_to_expr(expr)); }
        }
    }
    for (name, expr) in next_assigns.into_iter() {
        if let Some(idx) = name_to_idx.get(&name) {
            if *idx < var_count { next_exprs[*idx] = Some(pair_to_expr(expr)); }
        }
    }

    Ok(LeanModule { wires, var_count, init_exprs, next_exprs })
}
