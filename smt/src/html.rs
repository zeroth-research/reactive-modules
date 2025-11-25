use crate::itype::ArithOp;
use crate::{dtype::DType, itype::IType};
use base::{atom::Atom, module::Module, term::Term, wire::Wire};
use visual::html::{DescriptionContext, Descriptor};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;


pub struct SmtDescriptor {
    // Map from numeric wire index -> display name (e.g. "x0" or "x0'").
    wire_names: RefCell<HashMap<usize, String>>,
    // Cache the module wires so describe_input/describe_output can look up wire dtypes.
    module_wires: RefCell<Option<Wire<DType>>>,
    id_to_name: HashMap<usize, String>,
}

impl SmtDescriptor {
    pub fn new(module: &Module<DType, IType>) -> Self {
        let d = SmtDescriptor {
            wire_names: RefCell::new(HashMap::new()),
            module_wires: RefCell::new(None),
            id_to_name: HashMap::new(),
        };
        // Cache module wires for describe_input/describe_output
        *d.module_wires.borrow_mut() = Some(module.wire()[0].clone());

        // Build wire name map first so describe_wire_id can use it.
        d.populate_wire_names(module);

        d
    }

    pub fn get_name(&self, id: usize) -> Option<&str> {
        self.id_to_name.get(&id).and_then(|s| Some(s.as_str()))
    }

    fn wire_name(&self, id: usize) -> String {
        if let Some(name) = self.get_name(id) {
            return name.into();
        }
        format!("w{id}")
    }

    fn dump_atom_section(&self, atom: &Atom<DType, IType>, sec: &str, fmt: &HashMap<&str, &str>) -> String {
        let mut s = String::new();

        match sec {
            "init" => {
                for term in atom.init().iter() {
                    write!(s, "  ").unwrap();
                    writeln!(s, "{}", self.dump_term(term, fmt)).unwrap();
                }
            }
            "update" => {
                for term in atom.update().iter() {
                    write!(s, "  ").unwrap();
                    writeln!(s, "{}", self.dump_term(term, fmt)).unwrap();
                }
            }
            _ => panic!("Invalid section"),
        }

        s
    }

    fn dump_term(&self, term: &Term<DType, IType>, fmt: &HashMap<&str, &str>) -> String {
        let empty_str = "";
        let fmt_emph = fmt.get("EMPH_START").unwrap_or(&empty_str);
        let fmt_emph_end = fmt.get("EMPH_END").unwrap_or(&empty_str);
        let color_term_lhs = fmt.get("COLOR_TERM_LHS").unwrap_or(&empty_str);
        let color_term_op = fmt.get("COLOR_TERM_OP").unwrap_or(&empty_str);
        let color_clr = fmt.get("COLOR_CLEAR").unwrap_or(&empty_str);

        let reads = term
            .reads()
            .iter()
            .map(|(id, _)| self.wire_name(id))
            .collect::<Vec<String>>()
            .join(", ");

        let writes = term
            .writes()
            .iter()
            .map(|(id, _)| self.wire_name(id))
            .collect::<Vec<String>>()
            .join(", ");

        format!(
            "{color_term_lhs}{writes}{color_clr} = {color_term_op}{fmt_emph}{}{fmt_emph_end}{color_clr}({reads})",
            term.itype()
        )
    }

    // TODO: improve with term context
    fn describe_term_flow_with_context(
        &self,
        term: &Term<DType, IType>,
        _atom_terms: &[Term<DType, IType>],
    ) -> String {
        format!("Term: {}", term)
    }

    fn populate_wire_names(&self, module: &Module<DType, IType>) {
        let pair = module.wire();
        let latched = &pair[0];
        let next = &pair[1];

        let mut map = self.wire_names.borrow_mut();
        map.clear();

        // Assign names by position in the latched vector: x0, x1, ...
        for (pos, (idx, _dtype)) in latched.iter().enumerate() {
            map.insert(idx, format!("x{}", pos));
        }

        // For the next wires, assign xN' for the corresponding position
        for (pos, (nidx, _dtype)) in next.iter().enumerate() {
            map.insert(nidx, format!("x{}'", pos));
        }
    }
}

impl Descriptor<DType, IType> for SmtDescriptor {
    fn describe_module(&self, module: &Module<DType, IType>, _how: DescriptionContext) -> String {
        // Describe the module with its variables classified into prvt, intf, extl
        let prvt = module.prvt()[0]
            .iter()
            .map(|(wire_id, _)| self.describe_wire_id(wire_id, DescriptionContext::Inline))
            .collect::<Vec<String>>();
        let intf = module.intf()[0]
            .iter()
            .map(|(wire_id, _)| self.describe_wire_id(wire_id, DescriptionContext::Inline))
            .collect::<Vec<String>>();
        let extl = module.extl()[0]
            .iter()
            .map(|(wire_id, _)| self.describe_wire_id(wire_id, DescriptionContext::Inline))
            .collect::<Vec<String>>();

        // Now describe each atom in the module
        let mut atoms_html = String::new();
        for (atom_id, atom) in module.atoms().iter().enumerate() {
            atoms_html.push_str(&format!(
                "<hr><h3>Atom {atom_id}</h3><div>{}</div>",
                self.describe_atom(atom, DescriptionContext::Inline)
            ));
        }
        
        // Combine module variable diagram and atoms description
        format!(
            "<h2>Module</h2>{}<h2>Atoms</h2>{}",
            module_variables_diagram(&prvt, &intf, &extl),
            atoms_html
        )
    }

    fn describe_atom(&self, atom: &Atom<DType, IType>, how: DescriptionContext) -> String {
        let mut html = String::new();

        // Header
        if matches!(how, DescriptionContext::Standalone) {
            html.push_str("<h2>Atom</h2>\n");
        }

        // Name
        let ctrl_wires: Vec<String> = atom
            .ctrl()
            .iter()
            .map(|(w, _)| self.describe_wire_id(w, DescriptionContext::Inline))
            .collect();
        let wait_wires: Vec<String> = atom
            .wait()
            .iter()
            .map(|(w, _)| self.describe_wire_id(w, DescriptionContext::Inline))
            .collect();
        let read_wires: Vec<String> = atom
            .read()
            .iter()
            .map(|(w, _)| self.describe_wire_id(w, DescriptionContext::Inline))
            .collect();

        // Helper to format wire lists
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

        let next_wires: HashSet<usize> = atom.ctrl().iter().map(|p| p.0).collect();
        for term in atom.init() {
            // consider only output terms (the rest will be shown by
            // `describe_term_flow_with_context`)
            if term.writes().iter().any(|w| next_wires.contains(&w.0)) {
                html.push_str(
                    self.describe_term_flow_with_context(term, atom.init())
                        .as_str(),
                );
                html.push_str("</br>\n");
            }
        }

        html.push_str("<h4>Update</h4>");
        for term in atom.update() {
            if term.writes().iter().any(|w| next_wires.contains(&w.0)) {
                html.push_str(
                    self.describe_term_flow_with_context(term, atom.update())
                        .as_str(),
                );
                html.push_str("</br>\n");
            }
        }

        html

    }

    fn describe_atom_section(&self, atom: &Atom<DType, IType>, sec: &str, how: DescriptionContext) -> String {
        if matches!(how, DescriptionContext::Node) {
            return match sec {
                "init" => "Init".into(),
                "update" => "Update".into(),
                _ => panic!("Invalid section"),
            };
        }

        let fmt = HashMap::from([
            ("BOLD_START", "<b>"),
            ("BOLD_END", "</b>"),
            ("COLOR_TERM_LHS", "<span style=\"color: #106EE2\">"),
            ("COLOR_TERM_OP", "<span style=\"color: #E88914\">"),
            ("COLOR_CLEAR", "</span>"),
        ]);
        format!(
            "<h2>Atom {}</h2><pre>\n{}</pre>",
            sec,
            self.dump_atom_section(atom, sec, &fmt)
        )
    }

    fn describe_term(&self, term: &Term<DType, IType>, how: DescriptionContext) -> String {
        if matches!(how, DescriptionContext::Node) {
            return term.itype().to_string();
        }

        match term.itype() {
            IType::Const(val) => {
                return format!("<h3>Constant</h3><p><code>{}</code> → <code>w{}</code></p><hr>",
                    val,
                    term.writes().iter().next().map(|(idx, _)| idx).unwrap()
                );
            }
            IType::Arith(ArithOp::Add) => {
                return format!("<h3>Addition</h3><p><code>w{}</code> + <code>w{}</code> → <code>w{}</code></p><hr>",
                    term.reads().iter().next().map(|(idx, _)| idx).unwrap(),
                    term.reads().iter().nth(1).map(|(idx, _)| idx).unwrap(),
                    term.writes().iter().next().map(|(idx, _)| idx).unwrap()
                );
            }
            _ => format!("<h3>{}</h3><p><code>{}</code> → <code>{}</code></p><hr>", term.itype(), term.reads(), term.writes()),
        }
    }
}


/// Generate an SVG diagram showing the module variable classification
/// into private (prvt), interface (intf), and external (extl) variables.
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