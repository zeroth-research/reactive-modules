use crate::primitives::action::Action;
use crate::primitives::atom::Atom;
use crate::primitives::module::Module;
use crate::primitives::variable::Variable;
use std::marker::PhantomData;

pub trait State<Var: Variable, Val: Value>: Default {
    fn get(&self, var: &Var) -> Option<&Val>;
    fn insert(&mut self, var: Var, val: Val);
    fn remove(&mut self, var: &Var) -> Option<Val>;
}

pub trait Value: Default {}

pub trait InterpretableAction<Var, Val, S>: Action<Var, S>
where
    Var: Variable,
    Val: Value,
    for<'a> &'a S: IntoIterator<Item = &'a Var>,
    for<'a> <&'a S as IntoIterator>::IntoIter: ExactSizeIterator,
{
    fn interpret<St: State<Var, Val>>(&self, state: &mut St);
}

pub struct Intepreter<
    Var: Variable,
    Val: Value,
    St: State<Var, Val>,
    Seq,
    In: InterpretableAction<Var, Val, Seq>,
    Up: InterpretableAction<Var, Val, Seq>,
> where
    for<'a> &'a Seq: IntoIterator<Item = &'a Var>,
    for<'a> <&'a Seq as IntoIterator>::IntoIter: ExactSizeIterator,
{
    module: Module<Var, Seq, In, Up>,
    // TODO: Wrap state and only allow modification of external variables outside this struct.
    state: St,
    val: PhantomData<Val>,
}

impl<
    Var: Variable,
    Val: Value,
    St: State<Var, Val>,
    Seq,
    In: InterpretableAction<Var, Val, Seq>,
    Up: InterpretableAction<Var, Val, Seq>,
> Intepreter<Var, Val, St, Seq, In, Up>
where
    for<'a> &'a Seq: IntoIterator<Item = &'a Var>,
    for<'a> <&'a Seq as IntoIterator>::IntoIter: ExactSizeIterator,
{
    fn call_action(&mut self, action: fn(atom: &Atom<Var, Seq, In, Up>, &mut St)) {
        #[cfg(debug_assertions)]
        {
            // All the external next variables must be initialized.
            debug_assert!(self.module.extl.into_iter().all(|v| {
                self.state
                    .get(&v.next().expect("Next variable needs to exist"))
                    .is_some()
            }));
            // All controlled next variables must not be initialized.
            debug_assert!(self.module.ctr_iter().all(|v| {
                self.state
                    .get(&v.next().expect("Next variable needs to exist"))
                    .is_none()
            }));
        }

        // Execute each atom.
        for atom in self.module.atoms.iter() {
            // Check that the next control variables of the atom are not initialized.
            debug_assert!(atom.ctr.into_iter().all(|v| {
                self.state
                    .get(&v.next().expect("Next variable needs to exist"))
                    .is_none()
            }));
            debug_assert!(
                atom.read
                    .into_iter()
                    .all(|v| { self.state.get(&v).is_some() })
            );
            debug_assert!(atom.wait.into_iter().all(|v| {
                self.state
                    .get(&v.next().expect("Next variable needs to exist"))
                    .is_some()
            }));

            action(atom, &mut self.state);

            // Check that the next control variables of the atom are initialized.
            debug_assert!(atom.ctr.into_iter().all(|v| {
                self.state
                    .get(&v.next().expect("Next variable needs to exist"))
                    .is_some()
            }));
        }

        // All the values of the next need to go into the latched.
        // Then the next values need to be erased.
        for var in self.module.vars_iter() {
            let val = self
                .state
                .remove(&var.next().unwrap())
                .expect("Variable needs to exist");
            self.state.insert(var.clone(), val);
        }
        debug_assert!(self.module.vars_iter().all(|v| {
            self.state.get(&v).is_some()
                && self
                    .state
                    .get(&v.next().expect("Next variable needs to exist"))
                    .is_none()
        }));
    }
    fn init(&mut self) {
        #[cfg(debug_assertions)]
        {
            // All the latched variables must be not initialized.
            debug_assert!(
                self.module
                    .vars_iter()
                    .all(|v| self.state.get(&v).is_none())
            );
        }
        self.call_action(|atom, state| atom.init.interpret(state));
    }

    fn update(&mut self) {
        #[cfg(debug_assertions)]
        {
            // All the latched variables must be initialized.
            debug_assert!(
                self.module
                    .vars_iter()
                    .all(|v| self.state.get(&v).is_some())
            );
        }
        self.call_action(|atom, state| atom.update.interpret(state));
    }
}

impl<
    Var: Variable,
    Val: Value,
    St: State<Var, Val>,
    Seq,
    In: InterpretableAction<Var, Val, Seq>,
    Up: InterpretableAction<Var, Val, Seq>,
> Iterator for Intepreter<Var, Val, St, Seq, In, Up>
where
    for<'a> &'a Seq: IntoIterator<Item = &'a Var>,
    for<'a> <&'a Seq as IntoIterator>::IntoIter: ExactSizeIterator,
{
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        self.update();
        Some(())
    }
}
