use base::{Atom, Module, Term};

mod writetohtml;

pub use writetohtml::{write_to_html, module_to_live_json};

/// Specifies in which context an element (like Wire, Term or Atom) are shown in the HTML page
pub enum DescriptionContext {
    Standalone, // the element is being displayed alone (probably in the right pan)
    Inline,     // the element is being displayed inline as a part of another element
    Edge,       // the element is being displayed on an edge
    Node,       // the element is being displayed as a node label
}

///
/// By implementing this trait, one can adjust how information
/// about terms, atoms, modules, etc. is dumped to HTML.
pub trait Descriptor<T, I> {
    fn describe_module(&self, module: &Module<T, I>, how: DescriptionContext) -> String;
    fn describe_atom(&self, atom: &Atom<T, I>, how: DescriptionContext) -> String;
    fn describe_atom_section(
        &self,
        atom: &Atom<T, I>,
        sec: &str,
        how: DescriptionContext,
    ) -> String;
    fn describe_term(&self, term: &Term<T, I>, how: DescriptionContext) -> String;

    fn describe_wire(&self, id: usize, how: DescriptionContext) -> String {
        self.describe_wire_id(id, how)
    }

    fn describe_wire_id(&self, id: usize, _how: DescriptionContext) -> String {
        format!("w{id}")
    }

    // describe node representing output. `id` is the identifier
    // of the input wire
    fn describe_input(&self, _id: usize) -> String {
        "(input node)".into()
    }
    // describe node representing output. `id` is the identifier
    // of the output wire
    fn describe_output(&self, _id: usize) -> String {
        "(output node)".into()
    }
}

///
/// This is the default Descriptor, it simply calls `display()`
/// on items.
struct DefaultDescriptor {}
impl<T, I> Descriptor<T, I> for DefaultDescriptor
where
    Module<T, I>: std::fmt::Display,
    Atom<T, I>: std::fmt::Display,
    Term<T, I>: std::fmt::Display,
    I: std::fmt::Display,
{
    fn describe_module(&self, module: &Module<T, I>, _how: DescriptionContext) -> String {
        module.to_string()
    }

    fn describe_atom(&self, atom: &Atom<T, I>, _how: DescriptionContext) -> String {
        atom.to_string()
    }

    fn describe_atom_section(
        &self,
        _atom: &Atom<T, I>,
        sec: &str,
        _how: DescriptionContext,
    ) -> String {
        format!("Atom {}", sec)
    }

    fn describe_term(&self, term: &Term<T, I>, _how: DescriptionContext) -> String {
        term.to_string()
    }
}
