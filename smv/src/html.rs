use crate::{dtype::DType, itype::IType};
use base::{atom::Atom, module::Module, term::{Block, Term}, wire::Wire};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use visual::html::module::{DescriptionContext, Descriptor};

/// A small, SMV-specific HTML descriptor used by the visualiser.
///
/// This implements the `visual::html::Descriptor` trait for the
/// SMV `DType`/`IType` pair and provides a compact HTML rendering of
/// modules, atoms and terms. It also builds a wire-name mapping so
/// wires are displayed as `x0, x1, ...` and next wires as `x0', x1', ...`.
pub struct SmvDescriptor {
    /// Map from numeric wire index -> display name (e.g. "x0" or "x0'").
    wire_names: RefCell<HashMap<usize, String>>,
    /// Cache the module wires so describe_input/describe_output can look up wire dtypes.
    module_wires: RefCell<HashMap<usize, DType>>,
}

impl SmvDescriptor {
    pub fn new(module: &Module<DType, IType>) -> Self {
        let mut wire_map = HashMap::new();
        // Cache all wire dtypes from both latched and next
        for wire in module.obs().wires() {
            wire_map.insert(wire.id(), *wire.dtype());
        }
        
        let d = SmvDescriptor {
            wire_names: RefCell::new(HashMap::new()),
            module_wires: RefCell::new(wire_map),
        };

        // Build wire name map first so describe_wire_id can use it.
        d.populate_wire_names(module);

        d
    }

    fn describe_term_flow_with_context(
        &self,
        term: &Term<DType, IType>,
        atom_terms: &Block<DType, IType>,
    ) -> String {
        use crate::itype::IType;

        // Build map from wire -> term that writes it (within this atom)
        let mut writer_map: std::collections::HashMap<usize, &Term<DType, IType>> =
            std::collections::HashMap::new();
        for t in atom_terms.iter() {
            for wire in t.write().wires() {
                writer_map.insert(wire.id(), t);
            }
        }

        // Render a wire: if it has a non-user name (xN/xN') return it,
        // otherwise if it's an internal wN and has a writer, expand it.
        fn render_wire(
            rec: &SmvDescriptor,
            idx: usize,
            writer_map: &std::collections::HashMap<usize, &Term<DType, IType>>,
            visited: &mut HashSet<usize>,
        ) -> String {
            let name = rec.describe_wire_id(idx, DescriptionContext::Inline);
            if !name.starts_with('w') {
                return format!("<code>{}</code>", name);
            }
            if visited.contains(&idx) {
                return format!("<code>{}</code>", name);
            }
            if let Some(t) = writer_map.get(&idx) {
                visited.insert(idx);
                let s = render_term_expr(rec, t, writer_map, visited);
                visited.remove(&idx);
                return s;
            }
            format!("<code>{}</code>", name)
        }

        // Render a term as an expression (without its target arrow)
        fn render_term_expr(
            rec: &SmvDescriptor,
            term: &Term<DType, IType>,
            writer_map: &std::collections::HashMap<usize, &Term<DType, IType>>,
            visited: &mut HashSet<usize>,
        ) -> String {
            use crate::itype::IType;
            let it = term.itype();
            match it {
                IType::ConstInt(v) => format!("<code>{}</code>", v),
                IType::ConstBool(b) => format!("<code>{}</code>", b),
                IType::Add | IType::Sub | IType::Mul | IType::Div | IType::Mod => {
                    let rds: Vec<_> = term.read().iter().collect();
                    if rds.len() >= 2 {
                        let op = match it {
                            IType::Add => "+",
                            IType::Sub => "-",
                            IType::Mul => "*",
                            IType::Div => "/",
                            IType::Mod => "mod",
                            _ => "?",
                        };
                        let a = render_wire(rec, rds[0][0].id(), writer_map, visited);
                        let b = render_wire(rec, rds[1][0].id(), writer_map, visited);
                        return format!("{} {} {}", a, op, b);
                    }
                    String::new()
                }
                IType::Neg => {
                    let rds: Vec<_> = term.read().iter().collect();
                    if let Some([wire]) = rds.first() {
                        let s = render_wire(rec, wire.id(), writer_map, visited);
                        return format!("-{}", s);
                    }
                    String::new()
                }
                IType::Not => {
                    let rds: Vec<_> = term.read().iter().collect();
                    if let Some([wire]) = rds.first() {
                        let s = render_wire(rec, wire.id(), writer_map, visited);
                        return format!("!{}", s);
                    }
                    String::new()
                }
                IType::And | IType::Or | IType::Xor | IType::Xnor | IType::Implies => {
                    let rds: Vec<_> = term.read().iter().collect();
                    if rds.len() >= 2 {
                        let op = match it {
                            IType::And => "∧",
                            IType::Or => "∨",
                            IType::Xor => "xor",
                            IType::Xnor => "↔",
                            IType::Implies => "→",
                            _ => "?",
                        };
                        let a = render_wire(rec, rds[0][0].id(), writer_map, visited);
                        let b = render_wire(rec, rds[1][0].id(), writer_map, visited);
                        return format!("{} {} {}", a, op, b);
                    }
                    String::new()
                }
                IType::Lt | IType::Le | IType::Gt | IType::Ge | IType::Eq | IType::Neq => {
                    let rds: Vec<_> = term.read().iter().collect();
                    if rds.len() >= 2 {
                        let op = match it {
                            IType::Lt => "<",
                            IType::Le => "<=",
                            IType::Gt => ">",
                            IType::Ge => ">=",
                            IType::Eq => "==",
                            IType::Neq => "!=",
                            _ => "?",
                        };
                        let a = render_wire(rec, rds[0][0].id(), writer_map, visited);
                        let b = render_wire(rec, rds[1][0].id(), writer_map, visited);
                        return format!("{} {} {}", a, op, b);
                    }
                    String::new()
                }
                IType::Abs => {
                    let rds: Vec<_> = term.read().iter().collect();
                    if let Some([wire]) = rds.first() {
                        let s = render_wire(rec, wire.id(), writer_map, visited);
                        return format!("|{}|", s);
                    }
                    String::new()
                }
                IType::Cond => {
                    let rds: Vec<_> = term.read().iter().collect();
                    if rds.len() >= 3 {
                        let c = render_wire(rec, rds[0][0].id(), writer_map, visited);
                        let t = render_wire(rec, rds[1][0].id(), writer_map, visited);
                        let e = render_wire(rec, rds[2][0].id(), writer_map, visited);
                        // Use parentheses so complex sub-expressions are clearly grouped
                        return format!("({}) ? ({}) : ({})", c, t, e);
                    }
                    String::new()
                }
                IType::Next | IType::Assign | IType::Init => {
                    // simple pass-through of sources
                    let srcs: Vec<String> = term
                        .read()
                        .wires()
                        .map(|wire| render_wire(rec, wire.id(), writer_map, visited))
                        .collect();
                    srcs.join(", ")
                }
                IType::BitSelect(h, l) => {
                    let rds: Vec<_> = term.read().iter().collect();
                    if let Some([wire]) = rds.first() {
                        let s = render_wire(rec, wire.id(), writer_map, visited);
                        return format!("{}[{}:{}]", s, h, l);
                    }
                    String::new()
                }
                IType::Extend(n) => {
                    let rds: Vec<_> = term.read().iter().collect();
                    if let Some([wire]) = rds.first() {
                        let s = render_wire(rec, wire.id(), writer_map, visited);
                        return format!("extend({}, {})", s, n);
                    }
                    String::new()
                }
                IType::ToBool => {
                    let rds: Vec<_> = term.read().iter().collect();
                    if let Some([wire]) = rds.first() {
                        let s = render_wire(rec, wire.id(), writer_map, visited);
                        return format!("bool({})", s);
                    }
                    String::new()
                }
                IType::ToWord1 => {
                    let rds: Vec<_> = term.read().iter().collect();
                    if let Some([wire]) = rds.first() {
                        let s = render_wire(rec, wire.id(), writer_map, visited);
                        return format!("word1({})", s);
                    }
                    String::new()
                }
                IType::ToUnsigned => {
                    let rds: Vec<_> = term.read().iter().collect();
                    if let Some([wire]) = rds.first() {
                        let s = render_wire(rec, wire.id(), writer_map, visited);
                        return format!("unsigned({})", s);
                    }
                    String::new()
                }
                IType::ToSigned => {
                    let rds: Vec<_> = term.read().iter().collect();
                    if let Some([wire]) = rds.first() {
                        let s = render_wire(rec, wire.id(), writer_map, visited);
                        return format!("signed({})", s);
                    }
                    String::new()
                }
            }
        }

        // Top-level: render the expression and append the target(s) for this term
        let mut visited = HashSet::new();
        let expr = render_term_expr(self, term, &writer_map, &mut visited);
        if expr.is_empty() {
            return String::new();
        }
        // Use the first write that exists as the target label
        let write_wires: Vec<_> = term.write().wires().collect();
        if let Some(w) = write_wires.first() {
            let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
            return format!("{} → <code>{}</code>", expr, tgt);
        }
        expr
    }

    /// Populate the internal wire-name mapping from a module.
    fn populate_wire_names(&self, module: &Module<DType, IType>) {
        let obs = module.obs();
        let mut latched_wires: Vec<&Wire<DType>> = Vec::new();
        let mut next_wires: Vec<&Wire<DType>> = Vec::new();
        
        // obs.iter() returns [[&Wire; N]; M] where first is latched, second is next
        for (i, wires_array) in obs.iter().enumerate() {
            if i == 0 {
                latched_wires.extend(wires_array.iter().copied());
            } else if i == 1 {
                next_wires.extend(wires_array.iter().copied());
            }
        }

        let mut map = self.wire_names.borrow_mut();
        map.clear();

        // Assign names by position in the latched vector: x0, x1, ...
        for (pos, wire) in latched_wires.iter().enumerate() {
            map.insert(wire.id(), format!("x{}", pos));
        }

        // For the next wires, assign xN' for the corresponding position
        for (pos, wire) in next_wires.iter().enumerate() {
            map.insert(wire.id(), format!("x{}'", pos));
        }
    }

    fn describe_term_label(&self, term: &Term<DType, IType>) -> String {
        use crate::itype::IType;
        match term.itype() {
            IType::ConstInt(v) => format!("{}", v),
            IType::ConstBool(b) => format!("{}", b),
            other => other.to_string(),
        }
    }

    fn describe_term_flow(&self, term: &Term<DType, IType>) -> String {
        use crate::itype::IType;

        let it = term.itype();
        let writes: Vec<(usize, DType)> = term.write().wires().map(|w| (w.id(), *w.dtype())).collect();
        let reads: Vec<(usize, DType)> = term.read().wires().map(|w| (w.id(), *w.dtype())).collect();

        let emit_tgt = |widx: usize| {
            format!(
                "<code>{}</code>",
                self.describe_wire_id(widx, DescriptionContext::Inline)
            )
        };
        let emit_src = |r: &(usize, DType)| {
            format!(
                "<code>{}</code>",
                self.describe_wire_id(r.0, DescriptionContext::Inline)
            )
        };

        match it {
            IType::ConstInt(v) => {
                if let Some((widx, _)) = writes.first() {
                    format!("<code>{}</code> → {}", v, emit_tgt(*widx))
                } else {
                    format!("<code>{}</code>", v)
                }
            }
            IType::ConstBool(b) => {
                if let Some((widx, _)) = writes.first() {
                    format!("<code>{}</code> → {}", b, emit_tgt(*widx))
                } else {
                    format!("<code>{}</code>", b)
                }
            }
            IType::Add | IType::Sub | IType::Mul | IType::Div | IType::Mod => {
                if !reads.is_empty() && !writes.is_empty() && reads.len() >= 2 {
                    let op = match it {
                        IType::Add => "+",
                        IType::Sub => "-",
                        IType::Mul => "*",
                        IType::Div => "/",
                        IType::Mod => "mod",
                        _ => "?",
                    };
                    let src1 = emit_src(&reads[0]);
                    let src2 = emit_src(&reads[1]);
                    let tgt = emit_tgt(writes[0].0);
                    return format!("{} {} {} → {}", src1, op, src2, tgt);
                }
                String::new()
            }
            IType::Neg => {
                if !reads.is_empty() && !writes.is_empty() {
                    let src = emit_src(&reads[0]);
                    let tgt = emit_tgt(writes[0].0);
                    return format!("-{} → {}", src, tgt);
                }
                String::new()
            }
            IType::Not => {
                if !reads.is_empty() && !writes.is_empty() {
                    let src = emit_src(&reads[0]);
                    let tgt = emit_tgt(writes[0].0);
                    return format!("!{} → {}", src, tgt);
                }
                String::new()
            }
            IType::And | IType::Or | IType::Xor | IType::Xnor | IType::Implies => {
                if !reads.is_empty() && !writes.is_empty() && reads.len() >= 2 {
                    let op = match it {
                        IType::And => "∧",
                        IType::Or => "∨",
                        IType::Xor => "xor",
                        IType::Xnor => "↔",
                        IType::Implies => "→",
                        _ => "?",
                    };
                    let src1 = emit_src(&reads[0]);
                    let src2 = emit_src(&reads[1]);
                    let tgt = emit_tgt(writes[0].0);
                    return format!("{} {} {} → {}", src1, op, src2, tgt);
                }
                String::new()
            }
            IType::Lt | IType::Le | IType::Gt | IType::Ge | IType::Eq | IType::Neq => {
                if !reads.is_empty() && !writes.is_empty() && reads.len() >= 2 {
                    let op = match it {
                        IType::Lt => "<",
                        IType::Le => "<=",
                        IType::Gt => ">",
                        IType::Ge => ">=",
                        IType::Eq => "==",
                        IType::Neq => "!=",
                        _ => "?",
                    };
                    let src1 = emit_src(&reads[0]);
                    let src2 = emit_src(&reads[1]);
                    let tgt = emit_tgt(writes[0].0);
                    return format!("{} {} {} → {}", src1, op, src2, tgt);
                }
                String::new()
            }
            IType::Next | IType::Assign => {
                if !reads.is_empty() && !writes.is_empty() {
                    let srcs: Vec<String> = reads.iter().map(&emit_src).collect();
                    let src_join = srcs.join(", ");
                    let tgt = emit_tgt(writes[0].0);
                    return format!("{} → {}", src_join, tgt);
                }
                String::new()
            }
            IType::Init => {
                if !writes.is_empty() {
                    if !reads.is_empty() {
                        let srcs: Vec<String> = reads.iter().map(&emit_src).collect();
                        let src_join = srcs.join(", ");
                        let tgt = emit_tgt(writes[0].0);
                        return format!("{} → {}", src_join, tgt);
                    } else {
                        // e.g., a constant init produced as an init term
                        let tgt = emit_tgt(writes[0].0);
                        return tgt.to_string();
                    }
                }
                String::new()
            }
            IType::Abs => {
                if !reads.is_empty() && !writes.is_empty() {
                    let src = emit_src(&reads[0]);
                    let tgt = emit_tgt(writes[0].0);
                    return format!("|{}| → {}", src, tgt);
                }
                String::new()
            }
            IType::Cond => {
                if !writes.is_empty() && reads.len() >= 3 {
                    let c = emit_src(&reads[0]);
                    let t = emit_src(&reads[1]);
                    let e = emit_src(&reads[2]);
                    let tgt = emit_tgt(writes[0].0);
                    return format!("({}) ? ({}) : ({}) → {}", c, t, e, tgt);
                }
                String::new()
            }
            IType::BitSelect(h, l) => {
                if !reads.is_empty() && !writes.is_empty() {
                    let src = emit_src(&reads[0]);
                    let tgt = emit_tgt(writes[0].0);
                    return format!("{}[{}:{}] → {}", src, h, l, tgt);
                }
                String::new()
            }
            IType::Extend(n) => {
                if !reads.is_empty() && !writes.is_empty() {
                    let src = emit_src(&reads[0]);
                    let tgt = emit_tgt(writes[0].0);
                    return format!("extend({}, {}) → {}", src, n, tgt);
                }
                String::new()
            }
            IType::ToBool => {
                if !reads.is_empty() && !writes.is_empty() {
                    let src = emit_src(&reads[0]);
                    let tgt = emit_tgt(writes[0].0);
                    return format!("bool({}) → {}", src, tgt);
                }
                String::new()
            }
            IType::ToWord1 => {
                if !reads.is_empty() && !writes.is_empty() {
                    let src = emit_src(&reads[0]);
                    let tgt = emit_tgt(writes[0].0);
                    return format!("word1({}) → {}", src, tgt);
                }
                String::new()
            }
            IType::ToUnsigned => {
                if !reads.is_empty() && !writes.is_empty() {
                    let src = emit_src(&reads[0]);
                    let tgt = emit_tgt(writes[0].0);
                    return format!("unsigned({}) → {}", src, tgt);
                }
                String::new()
            }
            IType::ToSigned => {
                if !reads.is_empty() && !writes.is_empty() {
                    let src = emit_src(&reads[0]);
                    let tgt = emit_tgt(writes[0].0);
                    return format!("signed({}) → {}", src, tgt);
                }
                String::new()
            }
        }
    }

    fn describe_wire_label_for_edge(&self, id: usize) -> String {
        // Explicitly return the internal wire name for edge labels.
        format!("w{}", id)
    }
}

// FIXME: this is duplicated with the `toy` create
fn module_variables_diagram(prvt: &[String], intf: &[String], extl: &[String]) -> String {
    format!(
        r##"
        <div style="display:flex;flex-direction:column;align-items:center;margin:0;padding:0;">
        <svg width="100%" height="120" viewBox="0 0 600 120" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
            <!-- We'll split the vertical area above the wire row into two bands -->
            <!-- ctrl will be centered in the top band, obs centered in the middle band -->

                <!-- ctrl and obs side-by-side: split the horizontal space into two halves -->
                <rect x="40" y="-20" width="220" height="36" rx="8" fill="#fff2cc" stroke="#b08900"/>
                <text x="150" y="4" font-size="20" text-anchor="middle" fill="#111">ctrl</text>

                <rect x="340" y="-20" width="220" height="36" rx="8" fill="#e6ffe6" stroke="#2f8f3a"/>
                <text x="450" y="4" font-size="20" text-anchor="middle" fill="#111">obs</text>

            <!-- bottom boxes: prvt, intf, extl (aligned horizontally on the same baseline) -->
            <!-- prvt: pastel red, intf: pastel yellow, extl: pastel blue -->
            <rect x="60" y="96" width="140" height="50" rx="8" fill="#ffe5e5" stroke="#d16a6a"/>
            <text x="130" y="118" font-size="20" text-anchor="middle" fill="#111">prvt</text>
            <text x="130" y="134" font-size="16" text-anchor="middle" fill="#111">{}</text>
            <rect x="230" y="96" width="140" height="50" rx="8" fill="#fff8d6" stroke="#d1b35a"/>
            <text x="300" y="118" font-size="20" text-anchor="middle" fill="#111">intf</text>
            <text x="300" y="134" font-size="16" text-anchor="middle" fill="#111">{}</text>
            <rect x="400" y="96" width="140" height="50" rx="8" fill="#d0e7ff" stroke="#6b9bd1"/>
            <text x="470" y="118" font-size="20" text-anchor="middle" fill="#111">extl</text>
            <text x="470" y="134" font-size="16" text-anchor="middle" fill="#111">{}</text>
            <!-- arrows: from ctrl center (300,56) to prvt center (130,162) and intf center (300,162) -->
            <defs>
                <marker id="arrow" markerWidth="10" markerHeight="10" refX="10" refY="5" orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L10,5 L0,10 z" fill="#666"/>
                </marker>
            </defs>
                    <!-- arrows originate at bottom edge of ctrl (y=16) and hit top edge of targets (y=96) -->
                        <!-- orthogonal (axis-aligned) paths from ctrl bottom to targets -->
                        <path d="M150 16 V56 H130 V96" fill="none" stroke="#666" stroke-width="1.6" marker-end="url(#arrow)"/>
                            <!-- route to intf offset slightly left to avoid overlapping the obs->intf path -->
                            <path d="M150 16 V56 H285 V96" fill="none" stroke="#666" stroke-width="1.6" marker-end="url(#arrow)"/>
                <!-- arrows: from obs center (450,38) to intf center (300,162) and extl center (470,162) -->
                    <!-- arrows originate at bottom edge of obs (y=16) and hit top edge of targets (y=96) -->
                        <!-- orthogonal (axis-aligned) paths from obs bottom to targets -->
                            <!-- route to intf offset slightly right to avoid overlapping the ctrl->intf path -->
                            <path d="M450 16 V56 H315 V96" fill="none" stroke="#666" stroke-width="1.6" marker-end="url(#arrow)"/>
                        <path d="M450 16 V56 H470 V96" fill="none" stroke="#666" stroke-width="1.6" marker-end="url(#arrow)"/>
        </svg>
        </div>
    "##,
        prvt.join(", "),
        intf.join(", "),
        extl.join(", ")
    )
}

impl Descriptor<DType, IType> for SmvDescriptor {
    fn describe_module(&self, module: &Module<DType, IType>, _how: DescriptionContext) -> String {
        //let fmt = HashMap::from([("BOLD_START", "<b>"), ("BOLD_END", "</b>")]);
        //format!("<pre>\n{}</pre>", self.dump_module(module, &fmt))
        let prvt = module.prvt()
            .wires()
            .map(|wire| self.describe_wire_id(wire.id(), DescriptionContext::Inline))
            .collect::<Vec<String>>();
        let intf = module.intf()
            .wires()
            .map(|wire| self.describe_wire_id(wire.id(), DescriptionContext::Inline))
            .collect::<Vec<String>>();
        let extl = module.extl()
            .wires()
            .map(|wire| self.describe_wire_id(wire.id(), DescriptionContext::Inline))
            .collect::<Vec<String>>();

        // Append per-atom summaries: iterate over elements to find only
        // the top-level atom nodes (ids like `atom.<n>`). The element
        // list also contains the atom.init / atom.update cluster nodes
        // (which start with `atom.` as well) — we must avoid duplicating
        // the content by filtering those out.
        let mut atoms_html = String::new();
        for (atom_id, atom) in module.atoms().iter().enumerate() {
            atoms_html.push_str(&format!(
                "<hr><h3>Atom {atom_id}</h3><div>{}</div>",
                self.describe_atom(atom, DescriptionContext::Inline)
            ));
        }

        format!(
            "<h2>Module</h2>{}<h2>Atoms</h2>{}",
            module_variables_diagram(&prvt, &intf, &extl),
            atoms_html
        )
    }

    fn describe_atom(&self, atom: &Atom<DType, IType>, how: DescriptionContext) -> String {
        let mut html = String::new();

        if matches!(how, DescriptionContext::Standalone) {
            html.push_str("<h2>Atom</h2>\n");
        }

        // Build a small controls/waits/reads table for the atom using the
        // descriptor's wire naming so the atom panel shows the key ports.
        let ctrl_wires: Vec<String> = atom
            .ctrl()
            .wires()
            .map(|w| self.describe_wire_id(w.id(), DescriptionContext::Inline))
            .collect();
        let wait_wires: Vec<String> = atom
            .wait()
            .wires()
            .map(|w| self.describe_wire_id(w.id(), DescriptionContext::Inline))
            .collect();
        let read_wires: Vec<String> = atom
            .read()
            .wires()
            .map(|w| self.describe_wire_id(w.id(), DescriptionContext::Inline))
            .collect();

        let fmt_list = |v: Vec<String>| {
            if v.is_empty() {
                "<em>(none)</em>".into()
            } else {
                v.join(", ").to_string()
            }
        };

        let ctrl_html = fmt_list(ctrl_wires);
        let wait_html = fmt_list(wait_wires);
        let read_html = fmt_list(read_wires);

        html.push_str(format!(
            "<table><tr><td><strong>ctrl</strong></td><td>{}</td></tr><tr><td><strong>wait</strong></td><td>{}</td></tr><tr><td><strong>read</strong></td><td>{}</td></tr></table>",
            ctrl_html, wait_html, read_html
        ).as_str());

        html.push_str("<h4>Init</h4>");

        let next_wires: HashSet<usize> = atom.ctrl().wires().map(|w| w.id()).collect();
        for term in atom.init() {
            // consider only output terms (the rest will be shown by
            // `describe_term_flow_with_context`)
            if term.write().wires().any(|w| next_wires.contains(&w.id())) {
                html.push_str(
                    self.describe_term_flow_with_context(term, atom.init())
                        .as_str(),
                );
                html.push_str("</br>\n");
            }
        }

        html.push_str("<h4>Update</h4>");
        for term in atom.update() {
            if term.write().wires().any(|w| next_wires.contains(&w.id())) {
                html.push_str(
                    self.describe_term_flow_with_context(term, atom.update())
                        .as_str(),
                );
                html.push_str("</br>\n");
            }
        }

        html
    }

    fn describe_atom_section(
        &self,
        atom: &Atom<DType, IType>,
        sec: &str,
        how: DescriptionContext,
    ) -> String {
        let mut html = String::new();

        if matches!(how, DescriptionContext::Node) {
            return match sec {
                "init" => "Init".into(),
                "update" => "Update".into(),
                _ => format!("Unknown: {sec}"),
            };
        }

        let next_wires: HashSet<usize> = atom.ctrl().wires().map(|w| w.id()).collect();
        if sec == "init" {
            html.push_str("<h4>Init</h4>");

            for term in atom.init() {
                // consider only output terms (the rest will be shown by
                // `describe_term_flow_with_context`)
                if term.write().wires().any(|w| next_wires.contains(&w.id())) {
                    html.push_str(
                        self.describe_term_flow_with_context(term, atom.init())
                            .as_str(),
                    );
                    html.push_str("</br>\n");
                }
            }
        } else if sec == "update" {
            html.push_str("<h4>Update</h4>");
            for term in atom.update() {
                if term.write().wires().any(|w| next_wires.contains(&w.id())) {
                    html.push_str(
                        self.describe_term_flow_with_context(term, atom.update())
                            .as_str(),
                    );
                    html.push_str("</br>\n");
                }
            }
        } else {
            html.push_str(&format!("<h4>Unknown: {sec}</h4>"));
        }

        html
    }

    fn describe_term(&self, term: &Term<DType, IType>, how: DescriptionContext) -> String {
        if matches!(how, DescriptionContext::Node) {
            return self.describe_term_label(term);
        }

        if matches!(how, DescriptionContext::Inline) {
            return self.describe_term_flow(term);
        }

        use crate::itype::IType;

        let it = term.itype();

        // Build a simple reads/writes table to show wire names, ids and dtypes.
        let writes_v: Vec<_> = term.write().wires().collect();
        let reads_v: Vec<_> = term.read().wires().collect();

        let mut rows = String::new();
        rows.push_str("<table>");
        if !writes_v.is_empty() {
            for wire in writes_v.iter() {
                let name = self.describe_wire_id(wire.id(), DescriptionContext::Inline);
                rows.push_str(&format!(
					"<tr><td><strong>writes</strong></td><td><code>{}</code> <small>(w{} : {})</small></td></tr>",
					name, wire.id(), wire.dtype()
				));
            }
        }
        if !reads_v.is_empty() {
            for wire in reads_v.iter() {
                let name = self.describe_wire_id(wire.id(), DescriptionContext::Inline);
                rows.push_str(&format!(
					"<tr><td><strong>reads</strong></td><td><code>{}</code> <small>(w{} : {})</small></td></tr>",
					name, wire.id(), wire.dtype()
				));
            }
        }
        rows.push_str("</table>");

        // Try to provide a small, specific summary for certain instruction types
        let mut extra = String::new();
        let mut title_html = format!("<h3>{}</h3>", it);
        match it {
            IType::ConstInt(v) => {
                if let Some(w) = term.write().wires().next() {
                    let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                    title_html = "<h3>Constant (Int)</h3>".into();
                    extra = format!("<p><code>{}</code> → <code>{}</code></p>", v, tgt);
                }
            }
            IType::ConstBool(b) => {
                if let Some(w) = term.write().wires().next() {
                    let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                    title_html = "<h3>Constant (Bool)</h3>".into();
                    extra = format!("<p><code>{}</code> → <code>{}</code></p>", b, tgt);
                }
            }
            IType::Add | IType::Sub | IType::Mul | IType::Div | IType::Mod => {
                if let Some(w) = term.write().wires().next() {
                    let rds: Vec<_> = term.read().wires().collect();
                    if rds.len() >= 2 {
                        let op = match it {
                            IType::Add => "+",
                            IType::Sub => "-",
                            IType::Mul => "*",
                            IType::Div => "/",
                            IType::Mod => "mod",
                            _ => "?",
                        };
                        let src1 = self.describe_wire_id(rds[0].id(), DescriptionContext::Inline);
                        let src2 = self.describe_wire_id(rds[1].id(), DescriptionContext::Inline);
                        let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                        let title_str = match it {
                            IType::Add => "Addition",
                            IType::Sub => "Subtraction",
                            IType::Mul => "Multiplication",
                            IType::Div => "Division",
                            IType::Mod => "Modulo",
                            _ => "Arithmetic",
                        };
                        title_html = format!("<h3>{}</h3>", title_str);
                        extra = format!(
                            "<p><code>{}</code> {} <code>{}</code> → <code>{}</code></p>",
                            src1, op, src2, tgt
                        );
                    }
                }
            }
            IType::Neg => {
                if let (Some(w), Some(r)) = (term.write().wires().next(), term.read().wires().next())
                {
                    let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                    let src = self.describe_wire_id(r.id(), DescriptionContext::Inline);
                    title_html = "<h3>Negation</h3>".to_string();
                    extra = format!("<p><code>-{}</code> → <code>{}</code></p>", src, tgt);
                }
            }
            IType::Not => {
                if let (Some(w), Some(r)) = (term.write().wires().next(), term.read().wires().next())
                {
                    let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                    let src = self.describe_wire_id(r.id(), DescriptionContext::Inline);
                    extra = format!("<p><code>!{}</code> → <code>{}</code></p>", src, tgt);
                }
            }
            IType::And | IType::Or | IType::Xor | IType::Xnor | IType::Implies => {
                if let Some(w) = term.write().wires().next() {
                    let rds: Vec<_> = term.read().wires().collect();
                    if rds.len() >= 2 {
                        let op = match it {
                            IType::And => "∧",
                            IType::Or => "∨",
                            IType::Xor => "xor",
                            IType::Xnor => "↔",
                            IType::Implies => "→",
                            _ => "?",
                        };
                        let src1 = self.describe_wire_id(rds[0].id(), DescriptionContext::Inline);
                        let src2 = self.describe_wire_id(rds[1].id(), DescriptionContext::Inline);
                        let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                        extra = format!(
                            "<p><code>{}</code> {} <code>{}</code> → <code>{}</code></p>",
                            src1, op, src2, tgt
                        );
                    }
                }
            }
            IType::Lt | IType::Le | IType::Gt | IType::Ge | IType::Eq | IType::Neq => {
                if let Some(w) = term.write().wires().next() {
                    let rds: Vec<_> = term.read().wires().collect();
                    if rds.len() >= 2 {
                        let op = match it {
                            IType::Lt => "<",
                            IType::Le => "<=",
                            IType::Gt => ">",
                            IType::Ge => ">=",
                            IType::Eq => "==",
                            IType::Neq => "!=",
                            _ => "?",
                        };
                        let src1 = self.describe_wire_id(rds[0].id(), DescriptionContext::Inline);
                        let src2 = self.describe_wire_id(rds[1].id(), DescriptionContext::Inline);
                        let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                        let title_str = match it {
                            IType::Lt => "Less Than",
                            IType::Le => "Less Than or Equal",
                            IType::Gt => "Greater Than",
                            IType::Ge => "Greater Than or Equal",
                            IType::Eq => "Equality",
                            IType::Neq => "Not Equal",
                            _ => "Comparison",
                        };
                        title_html = format!("<h3>{}</h3>", title_str);
                        extra = format!(
                            "<p><code>{}</code> {} <code>{}</code> → <code>{}</code></p>",
                            src1, op, src2, tgt
                        );
                    }
                }
            }
            IType::Next => {
                if let (Some(w), Some(r)) = (term.write().wires().next(), term.read().wires().next())
                {
                    let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                    let src = self.describe_wire_id(r.id(), DescriptionContext::Inline);
                    extra = format!(
                        "<p><code>{}</code> → <code>{}</code> <small>(next)</small></p>",
                        src, tgt
                    );
                }
            }
            IType::Init => {
                let rds: Vec<_> = term.read().wires().collect();
                let writes_v: Vec<_> = term.write().wires().collect();
                if !writes_v.is_empty() {
                    let tgt = self.describe_wire_id(writes_v[0].id(), DescriptionContext::Inline);
                    if !rds.is_empty() {
                        let srcs: Vec<String> = rds
                            .iter()
                            .map(|r| {
                                format!(
                                    "<code>{}</code>",
                                    self.describe_wire_id(r.id(), DescriptionContext::Inline)
                                )
                            })
                            .collect();
                        let srcs_joined = srcs.join(", ");
                        extra = format!(
                            "<p>{} → <code>{}</code> <small>(init)</small></p>",
                            srcs_joined, tgt
                        );
                    } else {
                        extra = format!("<p><code>{}</code> <small>(init)</small></p>", tgt);
                    }
                }
            }
            IType::Assign => {
                if let (Some(w), Some(r)) = (term.write().wires().next(), term.read().wires().next())
                {
                    let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                    let src = self.describe_wire_id(r.id(), DescriptionContext::Inline);
                    extra = format!("<p><code>{}</code> → <code>{}</code></p>", src, tgt);
                }
            }
            IType::Abs => {
                if let (Some(w), Some(r)) = (term.write().wires().next(), term.read().wires().next())
                {
                    let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                    let src = self.describe_wire_id(r.id(), DescriptionContext::Inline);
                    title_html = "<h3>Absolute Value</h3>".to_string();
                    extra = format!("<p>|<code>{}</code>| → <code>{}</code></p>", src, tgt);
                }
            }
            IType::Cond => {
                let rds: Vec<_> = term.read().wires().collect();
                if let Some(w) = term.write().wires().next()
                    && rds.len() >= 3
                {
                    let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                    let c = self.describe_wire_id(rds[0].id(), DescriptionContext::Inline);
                    let t = self.describe_wire_id(rds[1].id(), DescriptionContext::Inline);
                    let e = self.describe_wire_id(rds[2].id(), DescriptionContext::Inline);
                    title_html = "<h3>Ternary Condition</h3>".to_string();
                    extra = format!(
                        "<p><code>{}</code> ? <code>{}</code> : <code>{}</code> → <code>{}</code></p>",
                        c, t, e, tgt
                    );
                }
            }
            IType::BitSelect(h, l) => {
                if let (Some(w), Some(r)) = (term.write().wires().next(), term.read().wires().next())
                {
                    let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                    let src = self.describe_wire_id(r.id(), DescriptionContext::Inline);
                    title_html = "<h3>Bit Select</h3>".to_string();
                    extra = format!("<p><code>{}</code>[{}:{}] → <code>{}</code></p>", src, h, l, tgt);
                }
            }
            IType::Extend(n) => {
                if let (Some(w), Some(r)) = (term.write().wires().next(), term.read().wires().next())
                {
                    let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                    let src = self.describe_wire_id(r.id(), DescriptionContext::Inline);
                    title_html = "<h3>Extend</h3>".to_string();
                    extra = format!("<p>extend(<code>{}</code>, {}) → <code>{}</code></p>", src, n, tgt);
                }
            }
            IType::ToBool => {
                if let (Some(w), Some(r)) = (term.write().wires().next(), term.read().wires().next())
                {
                    let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                    let src = self.describe_wire_id(r.id(), DescriptionContext::Inline);
                    title_html = "<h3>To Bool</h3>".to_string();
                    extra = format!("<p>bool(<code>{}</code>) → <code>{}</code></p>", src, tgt);
                }
            }
            IType::ToWord1 => {
                if let (Some(w), Some(r)) = (term.write().wires().next(), term.read().wires().next())
                {
                    let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                    let src = self.describe_wire_id(r.id(), DescriptionContext::Inline);
                    title_html = "<h3>To Word1</h3>".to_string();
                    extra = format!("<p>word1(<code>{}</code>) → <code>{}</code></p>", src, tgt);
                }
            }
            IType::ToUnsigned => {
                if let (Some(w), Some(r)) = (term.write().wires().next(), term.read().wires().next())
                {
                    let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                    let src = self.describe_wire_id(r.id(), DescriptionContext::Inline);
                    title_html = "<h3>To Unsigned</h3>".to_string();
                    extra = format!("<p>unsigned(<code>{}</code>) → <code>{}</code></p>", src, tgt);
                }
            }
            IType::ToSigned => {
                if let (Some(w), Some(r)) = (term.write().wires().next(), term.read().wires().next())
                {
                    let tgt = self.describe_wire_id(w.id(), DescriptionContext::Inline);
                    let src = self.describe_wire_id(r.id(), DescriptionContext::Inline);
                    title_html = "<h3>To Signed</h3>".to_string();
                    extra = format!("<p>signed(<code>{}</code>) → <code>{}</code></p>", src, tgt);
                }
            }
        }

        // Order: title, extra summary line, then reads/writes table.
        format!("{}{}<hr>{}", title_html, extra, rows)
    }

    fn describe_wire_id(&self, id: usize, how: DescriptionContext) -> String {
        if matches!(how, DescriptionContext::Edge) {
            return self.describe_wire_label_for_edge(id);
        }

        let map = self.wire_names.borrow();
        if let Some(name) = map.get(&id) {
            return name.clone();
        }

        format!("w{id}")
    }

    fn describe_input(&self, id: usize) -> String {
        let user_name = self.describe_wire_id(id, DescriptionContext::Inline);
        let dtype = self
            .module_wires
            .borrow()
            .get(&id)
            .copied()
            .unwrap_or(DType::Int);
        format!(
            "<h4>Input wire</h4>\n<code>{}</code> <small>(w{} : {})</small>",
            user_name, id, dtype
        )
    }

    fn describe_output(&self, id: usize) -> String {
        let user_name = self.describe_wire_id(id, DescriptionContext::Inline);
        let dtype = self
            .module_wires
            .borrow()
            .get(&id)
            .copied()
            .unwrap_or(DType::Int);
        format!(
            "<h4>Output wire</h4>\n<code>{}</code> <small>(w{} : {})</small>",
            user_name, id, dtype
        )
    }
}
