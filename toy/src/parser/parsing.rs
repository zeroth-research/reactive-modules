use pest::Parser as PestParser;
use pest::iterators::Pair;
use std::collections::HashMap;
use std::iter::zip;

use crate::context::Context;
use crate::dtype::Type;
use crate::instruction::{CmpOp, Instruction, LogicalOp};
use crate::term::*;
use crate::val::Val;

use base::atom::Atom;
use base::module::Module;
use base::term::Term;
use base::wire::Interface;

type ToyModule = Module<Type, Instruction>;
type ToyAtom = Atom<Type, Instruction>;
type ToyTerm = Term<Type, Instruction>;

#[derive(pest_derive::Parser)]
#[grammar = "parser/grammar.pest"]
pub struct ModuleParser;

#[derive(Debug)]
struct Var {
    name: String,
    _primed: bool,
}

pub struct Parser {
    ctx: Context,
}

impl Parser {
    pub fn new() -> Self {
        Parser {
            ctx: Context::new(),
        }
    }

    pub fn ctx(&self) -> &Context {
        &self.ctx
    }

    pub fn parse(&mut self, input: String) -> Vec<ToyModule> {
        let file = ModuleParser::parse(Rule::file, &input)
            .expect("Parsing failed")
            .next()
            .unwrap();

        self.parse_file(file)
    }

    fn parse_file(&mut self, pair: Pair<Rule>) -> Vec<ToyModule> {
        assert_eq!(pair.as_rule(), Rule::file);
        let mut modules = Vec::new();
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::modules {
                for m in inner.into_inner() {
                    modules.push(self.parse_module(m));
                }
            }
        }
        modules
    }

    fn parse_module(&mut self, pair: Pair<Rule>) -> Module<Type, Instruction> {
        // module = { "module" ~ ident ~ "{" ~ module_body ~ "}" }
        let mut inner = pair.into_inner();
        //let name = inner.next().unwrap().as_str().to_string();
        self.parse_module_body(inner.next().unwrap())
        //module.set_name(name.as_str());
    }

    fn parse_module_body(&mut self, pair: Pair<Rule>) -> Module<Type, Instruction> {
        // module_body   = { decl_variables ~ atoms }
        let mut inner = pair.into_inner();
        let decl_vars = parse_decl_variables(inner.next().unwrap());

        // add variables into the context
        let mut latched = Vec::new();
        let mut next = Vec::new();
        for (vars, ty) in decl_vars.values() {
            for v in vars {
                // latched variables
                latched.push(self.ctx.var(v.name.as_str(), *ty));
                // next variables
                next.push(self.ctx.var(format!("{}'", v.name).as_str(), *ty));
            }
        }

        let latched = Interface::sequence(latched.iter().copied()).unwrap();
        let next = Interface::sequence(next.iter().copied()).unwrap();

        let atoms_pair = inner.next().unwrap();
        let mut atoms = Vec::new();
        for a in atoms_pair.into_inner() {
            atoms.push(self.parse_atom(a, &[latched.clone(), next.clone()]));
        }

        ToyModule::observable(zip(latched, next).map(|([x], [y])| (x, y)), atoms).unwrap()
    }

    fn parse_atom(
        &mut self,
        pair: Pair<Rule>,
        module_wires: &[Interface<Type>; 2],
    ) -> Atom<Type, Instruction> {
        // atom = { "atom" ~ "{" ~ atom_body ~ "}" }
        let mut inner = pair.into_inner();
        let body = inner.next().unwrap();

        // atom_body = { atom_variables ~ atom_funs }
        let mut vars: HashMap<&str, Vec<Var>> = HashMap::new();

        let mut init: Vec<ToyTerm> = vec![];
        let mut update: Vec<ToyTerm> = vec![];
        for child in body.into_inner() {
            match child.as_rule() {
                // TODO: check no repetition
                Rule::atom_variables => {
                    for v in child.into_inner() {
                        match v.as_rule() {
                            Rule::atom_controls => {
                                vars.insert(
                                    "controls",
                                    parse_var_list(v.into_inner().next().unwrap()),
                                );
                            }
                            Rule::atom_reads => {
                                vars.insert(
                                    "reads",
                                    parse_var_list(v.into_inner().next().unwrap()),
                                );
                            }
                            Rule::atom_awaits => {
                                vars.insert(
                                    "awaits",
                                    parse_var_list(v.into_inner().next().unwrap()),
                                );
                            }
                            _ => {}
                        }
                    }
                }
                Rule::atom_funs => {
                    for f in child.into_inner() {
                        match f.as_rule() {
                            Rule::atom_init => {
                                init = self.parse_guarded_commands(f.into_inner().next().unwrap());
                            }
                            Rule::atom_update => {
                                update =
                                    self.parse_guarded_commands(f.into_inner().next().unwrap());
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        // TODO: check the inferred Interfaces with parsed spec for variables

        assert!(!update.is_empty());
        ToyAtom::sequential(
            module_wires[0].wires(),
            module_wires[1].wires(),
            init,
            update,
        )
        .unwrap()
    }

    fn parse_guarded_commands(&mut self, pair: Pair<Rule>) -> Vec<ToyTerm> {
        let mut terms: Vec<ToyTerm> = Vec::new();
        for g in pair.into_inner() {
            if g.as_rule() == Rule::guarded_command {
                let mut inner = g.into_inner();
                // guarded_command  = { "[]" ~ expr ~ "->" ~ assignments }
                let expr_pair = inner.next().unwrap();
                let guard = self.build_expr(expr_pair, Type::Bool);
                let assigns = self.parse_assignments(inner.next().unwrap());

                if guard.is_empty() {
                    dbg!("Failed creating a guard");
                    return vec![];
                }

                if assigns.is_empty() {
                    dbg!("Failed creating assignments");
                    return vec![];
                }

                let cond = guard.last().unwrap().write().clone();
                terms.extend(guard);
                for (var_name, expr) in assigns {
                    if expr.is_empty() {
                        panic!("Failed building assignment");
                    }

                    assert!(var_name.ends_with('\'')); // the assigned variable must be primed
                    let primed_var = self.ctx.get(var_name);
                    let var: Interface<Type> = self.ctx.get(&var_name[..var_name.len() - 1]);
                    let ite = mk_ite(
                        Interface::sequence(
                            [&cond, expr.last().unwrap().write(), &var]
                                .into_iter()
                                .flatten()
                                .map(|[i]| i.clone()),
                        )
                        .unwrap(),
                        primed_var,
                    )
                    .unwrap();
                    terms.extend(expr);
                    terms.push(ite);
                }
            }
        }
        terms
    }

    fn create_cmp_term(
        &mut self,
        op: &str,
        wire_l: &Interface<Type>,
        wire_r: &Interface<Type>,
    ) -> Vec<ToyTerm> {
        let mut negate = false;
        let op = match op {
            "<" | "\\lt" => CmpOp::Lt,
            ">" | "\\gt" => {
                negate = true;
                CmpOp::Le
            }
            "<=" | "\\le" => CmpOp::Le,
            ">=" | "\\ge" => {
                negate = true;
                CmpOp::Lt
            }
            "=" | "\\eq" => CmpOp::Eq,
            "!=" | "\\neq" => {
                negate = true;
                CmpOp::Eq
            }
            _ => unreachable!(),
        };
        let term = mk_cmp(
            op,
            Interface::sequence(wire_l.wires().chain(wire_r.wires()).cloned()).unwrap(),
            self.ctx.tmp_wire(Type::Bool),
        )
        .unwrap();

        if negate {
            let not = mk_not(term.write().clone(), self.ctx.tmp_wire(Type::Bool)).unwrap();
            return vec![term, not];
        }
        vec![term]
    }

    ///
    /// `ty` -- expected type of the resulting expression
    fn build_expr(&mut self, pair: Pair<Rule>, ty: Type) -> Vec<ToyTerm> {
        match pair.as_rule() {
            //              pair_a      op_       pair_b
            // expr  =   { expr_1 ~ (logic_op  ~ expr_1)* }
            // expr_1  = { expr_2 ~ (cmp_op  ~ expr_2)* }
            // expr_2  = { expr_atom ~ (arith_op  ~ expr_atom)* }
            Rule::expr | Rule::expr_1 | Rule::expr_2 => {
                let mut inner = pair.into_inner();
                let pair_a = inner.next().unwrap();
                // TODO: now we parse only integers, no support for floats at this moment
                //let ty = match rule {
                //    Rule::expr | Rule::expr_1 => Type::Bool,
                //    Rule::expr_2 => Type::Real,
                //    _ => ty,
                //};
                let mut terms = self.build_expr(pair_a, ty);
                let mut wire_l: Interface<Type>;

                while let Some(pair_b) = inner.next() {
                    // get the left-hand-side input wire to the term that we will parse next
                    // FIXME: make this more efficient
                    wire_l = terms.last().unwrap().write().clone();
                    if matches!(terms.last().unwrap().itype(), Instruction::Id) {
                        // the last created element was an Id instruction, so remove it
                        let last_term = terms.pop().unwrap();
                        wire_l = last_term.read().clone();
                    }

                    let op_ = pair_b.as_rule();
                    let op_str = pair_b.as_str();
                    let terms_b = self.build_expr(inner.next().unwrap(), ty);
                    if terms_b.is_empty() {
                        panic!("Couldn't build terms");
                    }
                    let mut wire_r = terms_b.last().unwrap().write().clone();
                    if matches!(terms_b.last().unwrap().itype(), Instruction::Id) {
                        wire_r = terms_b.last().unwrap().read().clone();
                    } else {
                        terms.extend(terms_b);
                    }

                    match op_ {
                        Rule::logic_op => {
                            let write_wire = self.ctx.tmp_wire(Type::Bool);
                            terms.push(
                                create_logic_term(op_str, &wire_l, &wire_r, write_wire).unwrap(),
                            )
                        }
                        Rule::arith_op => {
                            let (_, ty) = wire_l.wires().next().unwrap().into();
                            #[cfg(debug_assertions)]
                            {
                                let (_, ty2) = wire_r.wires().next().unwrap().into();
                                debug_assert!(ty == ty2);
                            }
                            let write_wire = self.ctx.tmp_wire(*ty);
                            terms.push(
                                create_arith_term(op_str, &wire_l, &wire_r, write_wire).unwrap(),
                            )
                        }
                        Rule::cmp_op => {
                            terms.extend(self.create_cmp_term(op_str, &wire_l, &wire_r))
                        }
                        _ => panic!("Invalid operation"),
                    }
                }
                terms
            }
            Rule::var | Rule::primed_var => {
                let var = pair.as_str();
                let (wire, ty) = self.ctx.get_with_type(var);
                let out = self.ctx.tmp_wire(ty);
                let term = mk_id(wire, out).unwrap();

                vec![term]
            }
            Rule::boolconst => match pair.as_str() {
                "true" => {
                    let out = self.ctx.tmp_wire(Type::Bool);
                    let term = mk_const(&Val::Bool(true), out);
                    match term {
                        Ok(t) => vec![t],
                        Err(e) => {
                            dbg!("ERROR: {}", e);
                            vec![]
                        }
                    }
                }
                "false" => {
                    let out = self.ctx.tmp_wire(Type::Bool);
                    let term = mk_const(&Val::Bool(false), out);
                    match term {
                        Ok(t) => vec![t],
                        Err(e) => {
                            dbg!("ERROR: {}", e);
                            vec![]
                        }
                    }
                }
                _ => unreachable!(),
            },
            Rule::constant => {
                let c = Val::from_str(pair.as_str(), ty).unwrap();
                vec![mk_const(&c, self.ctx.tmp_wire(ty)).unwrap()]
            }
            Rule::expr_atom => {
                // expr_atom = { (not_op | minus_op)? ~ expr_primary}
                let mut inner = pair.into_inner();
                let p = inner.next().unwrap();
                match p.as_rule() {
                    Rule::not_op => {
                        let mut expr = self.build_expr(inner.next().unwrap(), ty);
                        assert!(!expr.is_empty());
                        expr.push(
                            mk_not(
                                expr.last().unwrap().write().clone(),
                                self.ctx.tmp_wire(Type::Bool),
                            )
                            .unwrap(),
                        );
                        expr
                    }
                    Rule::minus_op => {
                        unimplemented!()
                    }
                    _ => self.build_expr(p, ty),
                }
            }
            _ => unreachable!("unexpected expr rule: {:?}", pair),
        }
    }

    fn parse_assignments<'a>(&mut self, pair: Pair<'a, Rule>) -> Vec<(&'a str, Vec<ToyTerm>)> {
        let mut terms = Vec::new();
        for p in pair.into_inner() {
            assert!(p.as_rule() == Rule::assign);
            let mut it = p.into_inner();
            // assign = { primed_var ~ ":=" ~ expr }
            let lhs = it.next().unwrap();
            let rhs = it.next().unwrap();
            assert!(lhs.as_rule() == Rule::primed_var);
            assert!(rhs.as_rule() == Rule::expr);

            let var = lhs.as_str();
            assert!(var.ends_with('\'')); // the variable must be primed
            let (_, ty) = self.ctx.get_with_type(var);
            let expr = self.build_expr(rhs, ty);
            if expr.is_empty() {
                dbg!("Failed creating an assignment");
                return vec![];
            }

            terms.push((var, expr));
        }
        terms
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

fn create_logic_term(
    op: &str,
    wire_l: &Interface<Type>,
    wire_r: &Interface<Type>,
    write: Interface<Type>,
) -> Result<ToyTerm, &'static str> {
    let op = match op {
        "&&" | "\\and" => LogicalOp::And,
        "||" | "\\or" => LogicalOp::Or,
        _ => unreachable!(),
    };
    mk_logical(
        op,
        Interface::from_iter([wire_l.clone(), wire_r.clone()].into_iter().flatten()),
        write,
    )
}

fn create_arith_term(
    op: &str,
    wire_l: &Interface<Type>,
    wire_r: &Interface<Type>,
    write: Interface<Type>,
) -> Result<ToyTerm, &'static str> {
    let args = [wire_l.clone(), wire_r.clone()].into_iter().flatten();
    match op {
        "+" => mk_add(Interface::from_iter(args), write),
        "-" => mk_sub(Interface::from_iter(args), write),
        "*" => mk_mul(Interface::from_iter(args), write),
        "/" => mk_div(Interface::from_iter(args), write),
        _ => Err("Invalid operation"),
    }
}

fn parse_decl_variables(pair: Pair<'_, Rule>) -> HashMap<&str, (Vec<Var>, Type)> {
    let mut vars: HashMap<&str, (Vec<Var>, Type)> = HashMap::new();

    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::ext_vars => {
                let _handle = vars.insert("ext", parse_var_block(p));
                debug_assert!(_handle.is_none());
            }
            Rule::intf_vars => {
                let _handle = vars.insert("intf", parse_var_block(p));
                debug_assert!(_handle.is_none());
            }
            _ => panic!("Invalid type of variables in module"),
        }
    }
    vars
}

fn parse_var_block(pair: Pair<Rule>) -> (Vec<Var>, Type) {
    // ext_vars = { "external" ~ var_list ~ ":" ~ type_ }
    // intf_vars = { "interface" ~ var_list ~ ":" ~ type_ }
    let mut inner = pair.into_inner();
    let vars = parse_var_list(inner.next().unwrap());
    let ty: Type = inner
        .next()
        .unwrap()
        .as_str()
        .parse()
        .expect("Invalid type");
    (vars, ty)
}

fn parse_var_list(pair: Pair<Rule>) -> Vec<Var> {
    pair.into_inner()
        .map(|p| {
            let mut name = p.as_str().to_string();
            let mut primed = false;
            if name.ends_with('\'') {
                name.pop();
                primed = true;
            }
            Var {
                name,
                _primed: primed,
            }
        })
        .collect()
}
