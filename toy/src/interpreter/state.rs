use crate::val::Val;
use std::collections::HashMap;

pub struct State {
    values: HashMap<usize, Val>,
}
