use pest::Parser as PestParser;
use pest::iterators::Pair;
use std::collections::{HashMap, HashSet};
use std::iter::zip;

use crate::ToyContext as Ctx;
use crate::itype::{CmpOp, LogicalOp};
use crate::term::*;
use crate::val::Val;
use crate::{DType, IType, ToyAtom, ToyModule, ToyTerm};

use base::module::Module;
use base::wire::Interface;

#[derive(pest_derive::Parser)]
#[grammar = "parser/grammar.pest"]
pub struct ModuleParser;

#[derive(Debug)]
struct Var {
    name: String,
    _primed: bool,
}

impl Var {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn primed_name(&self) -> String {
        debug_assert!(!self.name.ends_with('\''));
        format!("{}'", self.name)
    }
}

pub struct Parser {
    ctx: Ctx,
}

type StrErr = &'static str;

impl Parser {
    pub fn new() -> Self {
        Parser { ctx: Ctx::new() }
    }

    pub fn ctx(&self) -> &Ctx {
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

    fn parse_module(&mut self, pair: Pair<Rule>) -> Module<DType, IType> {
        // module = { "module" ~ ident ~ "{" ~ module_body ~ "}" }
        let mut inner = pair.into_inner();
        let _name = inner.next().unwrap().as_str().to_string();
        self.parse_module_body(inner.next().unwrap())
            .expect("Failed parsing a module")
        //module.set_name(name.as_str());
    }

    fn parse_module_body(&mut self, pair: Pair<Rule>) -> Result<Module<DType, IType>, StrErr> {
        // module_body   = { decl_variables ~ atoms }
        let mut inner = pair.into_inner();
        let decl_vars = parse_decl_variables(inner.next().unwrap());

        // add variables into the context
        let mut latched = Vec::new();
        let mut next = Vec::new();
        for (vars, ty) in decl_vars.values() {
            for v in vars {
                // latched variables
                latched.push(self.ctx.var(v.name(), *ty));
                // next variables
                next.push(self.ctx.var(v.primed_name().as_str(), *ty));
            }
        }

        let latched = Interface::sequence(latched.iter().copied()).unwrap();
        let next = Interface::sequence(next.iter().copied()).unwrap();

        let atoms_pair = inner.next().unwrap();
        let mut atoms = Vec::new();
        for a in atoms_pair.into_inner() {
            atoms.push(self.parse_atom(a, &[latched.clone(), next.clone()])?);
        }

        ToyModule::observable(zip(latched, next).map(|([x], [y])| (x, y)), atoms)
    }

    fn parse_atom(
        &mut self,
        pair: Pair<Rule>,
        module_wires: &[Interface<DType>; 2],
    ) -> Result<ToyAtom, StrErr> {
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
                                init = self.parse_guarded_commands(
                                    f.into_inner().next().unwrap(),
                                    vars["controls"].as_slice(),
                                );
                            }
                            Rule::atom_update => {
                                update = self.parse_guarded_commands(
                                    f.into_inner().next().unwrap(),
                                    vars["controls"].as_slice(),
                                );
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
    }

    fn parse_guarded_commands(&mut self, pair: Pair<Rule>, ctrl: &[Var]) -> Vec<ToyTerm> {
        let mut terms: Vec<ToyTerm> = Vec::new();
        // we need to group the expressions by the variable they define
        let mut exprs_per_var: HashMap<&str, Vec<Interface<DType>>> = HashMap::new();

        for g in pair.into_inner() {
            if g.as_rule() == Rule::guarded_command {
                let mut inner = g.into_inner();
                // guarded_command  = { "[]" ~ expr ~ "->" ~ assignments }
                let expr_pair = inner.next().unwrap();
                let guard = self.build_expr(expr_pair, DType::Bool);
                // assignments are a vector of (variable name, terms) meaning "assign the result of
                // terms into `name`"
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
                let mut assigned_variables: HashSet<&str> = HashSet::new();
                for (var_name, expr) in assigns {
                    debug_assert!(var_name.ends_with('\'')); // the assigned variable must be primed
                    if expr.is_empty() {
                        panic!("Failed building assignment");
                    }

                    if !assigned_variables.insert(var_name) {
                        panic!("Variable `{var_name}` assigned multiple times!")
                    }

                    let expr_out = expr.last().unwrap().write();
                    assert!(!expr_out.is_empty());
                    let fltr = mk_ifthen(
                        Interface::sequence(
                            [&cond, expr_out].into_iter().flatten().map(|[i]| i.clone()),
                        )
                        .unwrap(),
                        self.ctx.tmp_intf(*expr_out.wires().last().unwrap().dtype()),
                    )
                    .unwrap();

                    // this filter term is going to be input into a choose statement later
                    exprs_per_var
                        .entry(var_name)
                        .or_default()
                        .push(fltr.write().clone());

                    terms.extend(expr);
                    terms.push(fltr);
                }
            }
        }

        // add Choose terms -- they choose which assignemt is used
        for (var_name, inputs) in exprs_per_var.iter() {
            // get the write variable
            let primed_var = self.ctx.get_intf(var_name);
            // add the choose term
            let reads = Interface::sequence(
                inputs
                    .iter()
                    .flat_map(|i| {
                        debug_assert!(i.len() == 1);
                        i.wires()
                    })
                    .cloned(),
            )
            .unwrap();
            terms.push(construct(IType::Choose, reads, primed_var).unwrap());
        }

        // we have to write all ctrl variables, so for those that were not defined,
        // copy their current value
        for var in ctrl {
            let primed_name = var.primed_name();
            if exprs_per_var.contains_key(primed_name.as_str()) {
                continue;
            }

            let primed_var = self.ctx.get_intf(&primed_name);
            let var = self.ctx.get_intf(var.name());
            terms.push(construct(IType::Id, var, primed_var).unwrap());
        }
        terms
    }

    fn create_cmp_term(
        &mut self,
        op: &str,
        wire_l: &Interface<DType>,
        wire_r: &Interface<DType>,
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
            self.ctx.tmp_intf(DType::Bool),
        )
        .unwrap();

        if negate {
            let not = mk_not(term.write().clone(), self.ctx.tmp_intf(DType::Bool)).unwrap();
            return vec![term, not];
        }
        vec![term]
    }

    ///
    /// `ty` -- expected type of the resulting expression
    fn build_expr(&mut self, pair: Pair<Rule>, ty: DType) -> Vec<ToyTerm> {
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
                //    Rule::expr | Rule::expr_1 => DType::Bool,
                //    Rule::expr_2 => DType::Real,
                //    _ => ty,
                //};
                let mut terms = self.build_expr(pair_a, ty);
                let mut wire_l: Interface<DType>;

                while let Some(pair_b) = inner.next() {
                    // get the left-hand-side input wire to the term that we will parse next
                    // FIXME: make this more efficient
                    wire_l = terms.last().unwrap().write().clone();
                    if matches!(terms.last().unwrap().itype(), IType::Id) {
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
                    if matches!(terms_b.last().unwrap().itype(), IType::Id) {
                        wire_r = terms_b.last().unwrap().read().clone();
                    } else {
                        terms.extend(terms_b);
                    }

                    match op_ {
                        Rule::logic_op => {
                            let write_wire = self.ctx.tmp_intf(DType::Bool);
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
                            let write_wire = self.ctx.tmp_intf(*ty);
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
                let (wire, ty) = self.ctx.get_intf_with_type(var);
                let out = self.ctx.tmp_intf(ty);
                let term = mk_id(wire, out).unwrap();

                vec![term]
            }
            Rule::boolconst => match pair.as_str() {
                "true" => {
                    let out = self.ctx.tmp_intf(DType::Bool);
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
                    let out = self.ctx.tmp_intf(DType::Bool);
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
                vec![mk_const(&c, self.ctx.tmp_intf(ty)).unwrap()]
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
                                self.ctx.tmp_intf(DType::Bool),
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
            let (_, ty) = self.ctx.get_intf_with_type(var);
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
    wire_l: &Interface<DType>,
    wire_r: &Interface<DType>,
    write: Interface<DType>,
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
    wire_l: &Interface<DType>,
    wire_r: &Interface<DType>,
    write: Interface<DType>,
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

fn parse_decl_variables(pair: Pair<'_, Rule>) -> HashMap<&str, (Vec<Var>, DType)> {
    let mut vars: HashMap<&str, (Vec<Var>, DType)> = HashMap::new();

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

fn parse_var_block(pair: Pair<Rule>) -> (Vec<Var>, DType) {
    // ext_vars = { "external" ~ var_list ~ ":" ~ type_ }
    // intf_vars = { "interface" ~ var_list ~ ":" ~ type_ }
    let mut inner = pair.into_inner();
    let vars = parse_var_list(inner.next().unwrap());
    let ty: DType = inner
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
