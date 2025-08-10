use crate::primitives::action::Action;
use crate::primitives::atom::Atom;
use crate::primitives::module::Module;
use crate::primitives::variable::Variable;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
enum MyType {
    Integer,
    Real,
}

#[derive(Clone, Debug)]
struct Var {
    name: String,
    ty: MyType,
}

impl Var {
    fn new(name: String, ty: MyType) -> Self {
        Self { name, ty }
    }
}

impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Var {}

impl Hash for Var {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl Variable for Var {
    fn next(&self) -> Option<Self> {
        None
    }

    fn is_latched(&self) -> bool {
        true
    }
}

#[derive(Debug)]
enum Value {
    Integer(i64),
    Real(f64),
}

impl Value {
    fn test(ty: MyType) -> Self {
        match ty {
            MyType::Integer => Value::Integer(3),
            MyType::Real => Value::Real(2.5),
        }
    }
}

type State = HashMap<Var, Value>;

struct Update {
    read: Vec<Var>,
    write: Vec<Var>,
}
impl Action<Var, Vec<Var>> for Update {
    fn read(&self) -> &Vec<Var> {
        &self.read
    }

    fn write(&self) -> &Vec<Var> {
        &self.write
    }
}

impl Update {
    fn apply(&self, state: &mut State) {
        let mut acc: f64 = 0.0;
        for v in self.read.iter() {
            acc += match state.get(v).unwrap() {
                Value::Integer(i) => *i as f64,
                Value::Real(i) => *i,
            };
        }
        for v in self.write.iter() {
            state.insert(
                v.clone(),
                match v.ty {
                    MyType::Integer => Value::Integer(acc as i64),
                    MyType::Real => Value::Real(acc),
                },
            );
        }
    }
}

struct Init {
    read: Vec<Var>,
    write: Vec<Var>,
}
impl Action<Var, Vec<Var>> for Init {
    fn read(&self) -> &Vec<Var> {
        &self.read
    }

    fn write(&self) -> &Vec<Var> {
        &self.write
    }
}

impl Init {
    fn apply(&self, state: &mut State) {
        for v in self.write.iter() {
            state.insert(v.clone(), Value::test(v.ty));
        }
    }
}

pub fn interpret() {
    let x = Var::new("x".to_string(), MyType::Integer);
    let y = Var::new("y".to_string(), MyType::Real);
    let a = Atom::<Var, Vec<Var>, Init, Update>::new(
        vec![x.clone()],
        vec![x.clone()],
        vec![],
        Init {
            read: vec![],
            write: vec![x.clone()],
        },
        Update {
            read: vec![x.clone()],
            write: vec![x.clone()],
        },
    );
    let b = Atom::<Var, Vec<Var>, Init, Update>::new(
        vec![y.clone()],
        vec![x.clone()],
        vec![x.clone()],
        Init {
            read: vec![],
            write: vec![y.clone()],
        },
        Update {
            read: vec![x.clone()],
            write: vec![y.clone()],
        },
    );

    let m =
        Module::<Var, _, Init, Update>::new(vec![], vec![x.clone()], vec![y.clone()], vec![a, b]);

    let mut state = State::new();
    for a in m.atoms.iter() {
        a.init.apply(&mut state);
    }

    println!("{:?}", state);

    for a in m.atoms.iter() {
        a.update.apply(&mut state);
    }

    println!("{:?}", state);
}
